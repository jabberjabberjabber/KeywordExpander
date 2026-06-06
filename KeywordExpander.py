import json
import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field

import exiftool
import faiss
import numpy as np
import requests
from json_repair import repair_json as rj

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLORS = {
    'red', 'blue', 'green', 'yellow', 'purple', 'orange',
    'white', 'black', 'gray', 'grey', 'brown', 'beige',
    'pink', 'turquoise', 'golden', 'plaid', 'clear', 'metallic',
}

LLM_SYSTEM_PROMPT = """\
A JSON object will be returned which matches the given key as a word to a list of words that belong in a set with that key.

The KEY and CANDIDATES are provided as {word: [candidate, ...]}
The RESPONSE will be provided as {word: [verified candidate, ...]}

ONLY respond with candidates which you have VERIFIED.

How to verify a candidate:

- If the CANDIDATE means the same thing as the KEY, verify it
- If the KEY belongs to a SET for which the CANDIDATE is the PARENT, verify it

If no candidates match for verification, you may respond with an empty list after the key.

EXAMPLES:

VERIFY: KEY is "metal" and CANDIDATE is "material" since "metal" is a type of "material", making "material" the parent of a set which includes "metal".
IGNORE: KEY is "metal" and CANDIDATE is "brass" since "brass" is not a parent of "metal". "metal" cannot belong to a set where "brass" is the parent.

EXAMPLE INPUT: {"dog": ["poodle", "animal", "pet", "mammal", "canine"]}
EXAMPLE OUTPUT: {"dog": ["animal", "mammal", "canine"]}

EXAMPLE INPUT: {"serious": ["table", "crockpot", "funny", "brother"]}
EXAMPLE OUTPUT: {"serious": []}\

"""

class Config:
    def __init__(self):
        self.api_base_url = "http://localhost:8080" 
        self.api_key = ""
        self.embedding_model = ""
        self.chat_model = ""
        self.directory: Optional[str] = None
        self.max_candidates = 15
        self.similarity_threshold = 0.64
        self.skip_embeds = False
        self.skip_candidates = False
        self.load_json: Optional[str] = None

    @classmethod
    def from_args(cls, args) -> "Config":
        c = cls()
        c.api_base_url = args.api_base_url
        c.api_key = args.api_key
        c.embedding_model = args.embedding_model
        c.chat_model = args.chat_model
        c.directory = args.directory
        c.max_candidates = args.max_candidates
        c.similarity_threshold = args.similarity_threshold
        c.skip_embeds = args.skip_embeds
        c.skip_candidates = args.skip_candidates
        c.load_json = args.load_json
        return c

class OpenAIClient:
    def __init__(self, config: Config):
        base = config.api_base_url.rstrip("/")
        # Tolerate the user accidentally including /v1 in the base URL
        if base.endswith("/v1"):
            base = base[:-3]
        self.base_url = base
        self.embedding_model = config.embedding_model
        self.chat_model = config.chat_model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        })

    def get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            r = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.embedding_model, "input": text, "encoding_format": "float"},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed for '{text}': {e}")
            return None

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 200,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 120,
        min_p: float = 0.0,
        rep_pen: float = 1.05,
    ) -> Optional[str]:
        try:
            r = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "repetition_penalty": rep_pen,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            return None

@dataclass
class ProcessingStats:
    initial_keywords: int = 0
    unique_keywords: int = 0
    color_compounds_removed: int = 0
    compounds_split: int = 0
    final_keyword_count: int = 0
    embeddings_generated: int = 0
    candidate_pairs: int = 0
    verified_expansions: int = 0
    processing_times: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "\nProcessing Statistics:",
            f"  Initial keywords:        {self.initial_keywords}",
            f"  Unique keywords:         {self.unique_keywords}",
            f"  Color compounds removed: {self.color_compounds_removed}",
            f"  Compounds split:         {self.compounds_split}",
            f"  Final keyword set:       {self.final_keyword_count}",
            f"  Embeddings generated:    {self.embeddings_generated}",
            f"  Candidate pairs:         {self.candidate_pairs}",
            f"  Verified expansions:     {self.verified_expansions}",
        ]
        if self.processing_times:
            lines.append("\nTiming:")
            for step, t in self.processing_times.items():
                lines.append(f"  {step:<26} {t:.1f}s")
            lines.append(f"  {'total':<26} {sum(self.processing_times.values()):.1f}s")
        return "\n".join(lines)

class KeywordProcessor:
    """Three-step pipeline with checkpoints"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAIClient(config)
        self.stats = ProcessingStats()
        self._times: Dict[str, float] = {}
        self.keywords: List[str] = []
        self.candidate_mappings: Dict[str, List[str]] = {}
        self.expansions: Dict[str, List[str]] = {}

    def prepare(self, metadata: List[Dict]):
        """Extract keywords from metadata, clean, and split compounds"""
        t = time.time()
        raw = self._extract_raw_keywords(metadata)
        self.stats.initial_keywords = len(raw)

        cleaned = self._strip_color_prefixes(raw)

        unique = list(dict.fromkeys(cleaned))
        self.stats.unique_keywords = len(unique)

        self.keywords = self._split_compounds(unique)
        self.stats.final_keyword_count = len(self.keywords)

        self._times["preparation"] = time.time() - t
        self.stats.processing_times = dict(self._times)

    def embed(self, embeddings_file: Optional[Path] = None):
        """Generate embeddings and build FAISS candidate map"""
        t = time.time()
        self.candidate_mappings = self._build_candidate_map(embeddings_file)
        self._times["embedding"] = time.time() - t
        self.stats.processing_times = dict(self._times)

    def validate(self) -> Dict[str, List[str]]:
        """Validate candidates with LLM"""
        t = time.time()
        self.expansions = self._validate_candidates()
        self._times["llm_validation"] = time.time() - t
        self.stats.processing_times = dict(self._times)
        return self.expansions

    def _extract_raw_keywords(self, metadata: List[Dict]) -> List[str]:
        raw: List[str] = []
        for entry in metadata:
            kw = entry.get("Composite:Keywords") or entry.get("Subject")
            if not kw:
                continue
            if isinstance(kw, str):
                raw.append(kw)
            elif isinstance(kw, list):
                raw.extend(str(k) for k in kw)
        return raw

    def _strip_color_prefixes(self, keywords: List[str]) -> List[str]:
        """Replace 'red car' with 'car'"""
        result = []
        for kw in keywords:
            words = kw.split()
            if len(words) == 2 and words[0].lower() in COLORS:
                result.append(words[1])
                self.stats.color_compounds_removed += 1
            else:
                result.append(kw)
        return result

    def _find_splittable_modifiers(self, compounds: List[str]) -> set:
        """Return modifiers that appear in enough unique compounds to be descriptors.

        A modifier is splittable when it modifies many different bases,
        suggesting it is a standalone descriptor rather than part of a fixed phrase.
        Threshold: >3 occurrences, >70% unique compound ratio.
        """
        totals: Dict[str, int] = defaultdict(int)
        uniques: Dict[str, set] = defaultdict(set)
        for c in compounds:
            mod = c.split()[0]
            totals[mod] += 1
            uniques[mod].add(c)
        return {
            mod for mod, total in totals.items()
            if total > 3 and len(uniques[mod]) / total > 0.7
        }

    def _split_compounds(self, keywords: List[str]) -> List[str]:
        """Split compound keywords"""
        compounds = [kw for kw in keywords if 1 < len(kw.split()) < 4]
        splittable = self._find_splittable_modifiers(compounds)
        if not splittable:
            return keywords

        to_split = {c for c in compounds if c.split()[0] in splittable}
        survivors = [kw for kw in keywords if kw not in to_split]

        parts: List[str] = []
        for compound in to_split:
            self.stats.compounds_split += 1
            words = compound.split()
            if len(words) == 3 and words[1] in ("and", "or"):
                parts.extend([words[0], words[2]])
            elif len(words) == 2:
                parts.extend(words)

        return list(dict.fromkeys(survivors + parts))

    def _generate_embeddings(self) -> Optional[np.ndarray]:
        vectors = []
        n = len(self.keywords)
        for i, kw in enumerate(self.keywords):
            if i % 50 == 0:
                print(f"  Embedding {i}/{n}...")
            vec = self.client.get_embedding(kw)
            if vec is None:
                logger.warning(f"No embedding for '{kw}', substituting zeros")
                vec = [0.0] * 384
            vectors.append(vec)

        if not vectors:
            logger.error("No embeddings generated")
            return None

        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        return arr

    def _save_embeddings(self, arr: np.ndarray, path: Path):
        np.savez(str(path), embeddings=arr, keywords=np.array(self.keywords, dtype=object))
        logger.info(f"Saved embeddings checkpoint: {path}")

    def _load_embeddings(self, path: Path) -> Optional[np.ndarray]:
        try:
            data = np.load(str(path), allow_pickle=True)
            if data["keywords"].tolist() != self.keywords:
                logger.info("Embeddings checkpoint keyword list has changed — regenerating")
                return None
            logger.info(f"Loaded embeddings from {path}")
            return data["embeddings"].astype("float32")
        except Exception as e:
            logger.warning(f"Could not load embeddings checkpoint: {e}")
            return None

    def _build_candidate_map(self, embeddings_file: Optional[Path] = None) -> Dict[str, List[str]]:
        if not self.keywords:
            return {}

        embeddings = None
        if embeddings_file and embeddings_file.exists():
            embeddings = self._load_embeddings(embeddings_file)

        if embeddings is None:
            print(f"Generating embeddings for {len(self.keywords)} keywords...")
            embeddings = self._generate_embeddings()
            if embeddings is None:
                return {}
            if embeddings_file:
                self._save_embeddings(embeddings, embeddings_file)
                print(f"Embeddings checkpoint saved to {embeddings_file}")

        self.stats.embeddings_generated = len(self.keywords)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        candidates: Dict[str, List[str]] = {}
        for idx, tag in enumerate(self.keywords):
            if len(tag.split()) > 3:
                continue
            q = embeddings[idx].reshape(1, -1)
            dists, idxs = index.search(q, self.config.max_candidates + 1)
            matches = [
                self.keywords[i]
                for i, d in zip(idxs[0], dists[0])
                if self.keywords[i] != tag
                and d < self.config.similarity_threshold
                and len(self.keywords[i]) > 1
            ][: self.config.max_candidates]
            if matches:
                candidates[tag] = matches

        self.stats.candidate_pairs = sum(len(v) for v in candidates.values())
        return candidates

    @staticmethod
    def _parse_llm_response(raw: str) -> Optional[List[str]]:
        """Extract the candidate list from an LLM response.

        Returns [] for a valid but empty response, None when unparseable
        """
        def strings_from(data) -> Optional[List[str]]:
            """Return a flat list of strings from any parsed JSON structure
            """
            if isinstance(data, list):
                return [item for item in data if isinstance(item, str)]
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return [item for item in v if isinstance(item, str)]
                return []
            return None

        def try_parse(text: str) -> Optional[List[str]]:
            for loader in (json.loads, lambda t: json.loads(rj(t))):
                try:
                    result = strings_from(loader(text))
                    if result is not None:
                        return result
                except Exception:
                    pass
            return None

        if not raw:
            return None

        # strips <think> preamble and leading text
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            result = try_parse(raw[start : end + 1])
            if result is not None:
                return result

        # markdown fenced block ```json ... ```
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if m:
            result = try_parse(m.group(1))
            if result is not None:
                return result

        # array found anywhere in the text
        m = re.search(r"\[([^\[\]]*)\]", raw, re.DOTALL)
        if m:
            result = try_parse("[" + m.group(1) + "]")
            if result is not None:
                return result

        # repair the whole string
        return try_parse(raw)

    def _validate_candidates(self) -> Dict[str, List[str]]:
        if not self.candidate_mappings:
            return {}
        total = len(self.candidate_mappings)
        print(f"Validating {total} keywords with LLM...")
        expansions: Dict[str, List[str]] = {}
        candidate_set_cache: Dict[str, set] = {
            tag: set(cands) for tag, cands in self.candidate_mappings.items()
        }

        for i, (tag, candidates) in enumerate(self.candidate_mappings.items(), 1):
            if i % 10 == 0:
                print(f"  Validated {i}/{total}...")
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({tag: candidates}) + f'\nSTART GENERATION WITH {{"{tag}": '},
            ]
            raw = self.client.chat(messages)
            if not raw:
                continue
            try:
                result = self._parse_llm_response(raw)
                if result is None:
                    logger.warning(f"Unparseable LLM response for '{tag}': {raw[:80]!r}")
                    continue
                # reject any word not in the candidates list
                valid = [w for w in set(result) if w in candidate_set_cache[tag]]
                if valid:
                    expansions[tag] = valid
                    self.stats.verified_expansions += len(valid)
            except Exception as e:
                logger.error(f"Validation error for '{tag}': {e}")

        return expansions

def apply_expansions(
    metadata: List[Dict], expansions: Dict[str, List[str]]
) -> List[Dict]:
    """Return a copy of metadata entries with expanded keyword sets"""
    result = []
    for entry in metadata:
        if "Composite:Keywords" in entry:
            kw_key = "Composite:Keywords"
        elif "Subject" in entry:
            kw_key = "Subject"
        else:
            continue
        current = entry[kw_key]
        if isinstance(current, str):
            current = [current]
        expanded = set(current)
        for kw in set(current):
            if kw in expansions:
                expanded.update(expansions[kw])
        new_entry = entry.copy()
        new_entry[kw_key] = list(expanded)
        result.append(new_entry)
    return result


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved: {path}")

def main():
    parser = argparse.ArgumentParser(
        description="Expand image metadata keywords using semantic similarity and LLM validation"
    )
    parser.add_argument("directory", help="Directory containing image files")
    parser.add_argument(
        "--api-base-url", default="http://localhost:8080",
        help="Base URL for OpenAI-compatible API — do not include /v1 (default: %(default)s)",
    )
    parser.add_argument("--api-key", default="sk-no-key-required")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--chat-model", default="llama-3-8b-instruct")
    parser.add_argument(
        "--max-candidates", type=int, default=15,
        help="Max similar keywords to consider per keyword (default: %(default)s)",
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.64,
        help="FAISS L2 distance cutoff — lower is stricter (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-embeds", action="store_true",
        help="Skip embedding step and load existing candidates checkpoint from target directory",
    )
    parser.add_argument(
        "--skip-candidates", action="store_true",
        help="Skip LLM validation — stop after building the candidate map",
    )
    parser.add_argument(
        "--load-json", default=None,
        help="Load metadata from this JSON file instead of scanning the directory",
    )
    args = parser.parse_args()

    config = Config.from_args(args)
    target_dir = Path(config.directory)

    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        sys.exit(1)

    metadata_file   = target_dir / "KeywordExpander_metadata.json"
    embeddings_file = target_dir / "KeywordExpander_embeddings.npz"
    candidates_file = target_dir / "KeywordExpander_candidates.json"
    expansions_file = target_dir / "KeywordExpander_expansions.json"

    if config.load_json:
        print(f"Loading metadata from {config.load_json}")
        metadata = load_json(config.load_json)
    elif metadata_file.exists():
        print(f"Loading cached metadata from {metadata_file}")
        metadata = load_json(str(metadata_file))
    else:
        print(f"Extracting metadata from {target_dir}...")
        try:
            with exiftool.ExifToolHelper() as et:
                metadata = et.get_tags(str(target_dir), ["MWG:Keywords", "Subject"], "-r")
        except Exception as e:
            print(f"ExifTool error: {e}")
            sys.exit(1)
        if not metadata:
            print("No metadata found.")
            sys.exit(0)
        save_json(metadata, str(metadata_file))
        print(f"Metadata saved to {metadata_file}")

    processor = KeywordProcessor(config)
    processor.prepare(metadata)

    if config.skip_embeds:
        if candidates_file.exists():
            print(f"Loading cached candidates from {candidates_file}")
            processor.candidate_mappings = load_json(str(candidates_file))
        else:
            print("--skip-embeds set but no candidates checkpoint found. Run without --skip-embeds first.")
            sys.exit(1)
    else:
        processor.embed(embeddings_file)
        if processor.candidate_mappings:
            save_json(processor.candidate_mappings, str(candidates_file))
            print(f"Candidates checkpoint saved to {candidates_file}")

    print(processor.stats)

    if config.skip_candidates:
        print("--skip-candidates set; stopping after candidate map.")
        return

    if not processor.candidate_mappings:
        print("No candidates found. Try lowering --similarity-threshold.")
        return

    expansions = processor.validate()
    print(processor.stats)

    if not expansions:
        print("No expansions found.")
        return

    save_json(expansions, str(expansions_file))
    print(f"Expansions saved to {expansions_file}")

    expanded_metadata = apply_expansions(metadata, expansions)
    if not expanded_metadata:
        print("No entries to update.")
        return

    print(f"Writing expanded keywords to {len(expanded_metadata)} files...")
    success = fail = 0
    with exiftool.ExifToolHelper() as et:
        for entry in expanded_metadata:
            src = entry.get("SourceFile")
            if not src:
                continue
            file_path = Path(src)  # ExifTool returns absolute paths
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                fail += 1
                continue
            if not os.access(file_path, os.W_OK):
                logger.warning(f"No write permission: {file_path}")
                fail += 1
                continue
            kw_key = "Composite:Keywords" if "Composite:Keywords" in entry else "Subject"
            try:
                et.set_tags(
                    str(file_path),
                    tags={"MWG:Keywords": entry[kw_key]},
                    params=["-P", "-overwrite_original"],
                )
                success += 1
            except Exception as e:
                logger.error(f"Failed writing {file_path}: {e}")
                fail += 1

    print(f"Complete: {success} succeeded, {fail} failed.")


if __name__ == "__main__":
    main()
