import json
import exiftool
import numpy as np
import faiss
import subprocess
import os
import time
import sys
import argparse
from typing import List, Dict, Optional
from pathlib import Path
from koboldapi import KoboldAPICore
from json_repair import repair_json as rj
from dataclasses import dataclass
from collections import defaultdict

class Config:
    """ Configuration for keyword processing """
    def __init__(self):
        self.model_path = "all-MiniLM-L6-v2-ggml-model-f16.gguf"
        self.llama_path = "llama-embedding.exe"
        self.api_url = "http://localhost:5001"
        self.directory = None
        self.max_candidates = 15
        self.similarity_threshold = 0.64
        self.skip_candidates = False
        self.skip_embeds = False
        self.skip_metadata = False
        self.skip_compounds = False
        self.load_json = None
          
    @classmethod
    def from_args(cls, args):
        """ Create config from command line arguments """
        config = cls()
        config.model_path = args.model_path
        config.llama_path = args.llama_path
        config.api_url = args.api_url
        config.directory = args.directory

        return config
                     
class KeywordContainer:
    """ Contains and tracks state of keyword processing """
    def __init__(self, metadata: List[Dict]):
        self.file_metadata = metadata
        self.raw_keywords = None  # Initial keywords from metadata
        self.keywords = None      # Current set of keywords
        self.raw_compounds = None
        self.compounds = None     # Identified compound keywords
        self.singles = None       # Single word keywords
        self.removed = {}         # Tracks removed keywords and reasons
        self.candidate_mappings = None  # Dict[str, List[str]]
        self.keyword_expansions = None  # Dict[str, List[str]]
        self.compound_splits = None  # Dict[str, List[str]]

    @property 
    def has_keywords(self) -> bool:
        return bool(self.keywords)
        
    @property
    def has_compounds(self) -> bool:
        return bool(self.compounds)

    @property
    def has_compound_splits(self) -> bool:
        return bool(self.compound_splits)
    
    @property
    def has_candidates(self) -> bool:
        return bool(self.candidate_mappings)
        
    @property
    def has_expansions(self) -> bool:
        return bool(self.keyword_expansions)

    def prepare_output(self) -> List[Dict]:
        """ Creates new metadata entries with expanded keywords """
        result = []
        for entry in self.file_metadata:
            if 'Composite:Keywords' not in entry:
                continue
                
            new_entry = entry.copy()
            current_set = set(entry['Composite:Keywords'])
            for keyword in self.keyword_expansions:
                if keyword in current_set:
                    current_set.update(self.keyword_expansions[keyword])
            new_entry['Composite:Keywords'] = list(current_set)
            result.append(new_entry)
        return result
@dataclass
class ProcessingStats:
    """ Tracks statistics for keyword processing pipeline """
    initial_keywords: int = 0
    unique_keywords: int = 0
    single_keywords: int = 0
    unique_singles: int = 0
    compound_keywords: int = 0
    unique_compounds: int = 0
    unique_modifiers: int = 0
    unique_bases: int = 0
    split_compounds: int = 0
    color_compounds: int = 0
    embeddings_generated: int = 0
    candidate_pairs: int = 0
    verified_candidates: int = 0
    final_keywords: int = 0
    processing_times: dict = None
    
    def calculate_expansion_rate(self) -> float:
        """ Calculate the percentage increase in keywords """
        if self.initial_keywords == 0:
            return 0
        return ((self.final_keywords - self.initial_keywords) / 
                self.initial_keywords * 100)
    
    def __str__(self) -> str:
        """ Format stats for display """
        lines = [
            "Processing Statistics:",
            "\nInitial State:",
            f"  Total Keywords: {self.initial_keywords}",
            f"  Unique Keywords: {self.unique_keywords}",
            f"  Single Keywords: {self.single_keywords}",
            f"  Unique Singles: {self.unique_singles}",
            f"  Compound Keywords: {self.compound_keywords}",
            f"  Unique Compounds: {self.unique_compounds}",
            f"  Unique Modifiers: {self.unique_modifiers}",
            f"  Unique Bases: {self.unique_bases}",
            "\nProcessing Results:",
            f"  Split Compounds: {self.split_compounds}",
            f"  Color Compounds Split: {self.color_compounds}",
            f"  Embeddings Generated: {self.embeddings_generated}",
            f"  Candidates Found: {self.candidate_pairs}",
            f"  Verified Candidates: {self.verified_candidates}",
            f"  Final Total Keywords: {self.final_keywords}",
            f"  Expansion Rate: {self.calculate_expansion_rate():.1f}%"
        ]
        
        if self.processing_times:
            lines.extend([
                "\nProcessing Times:",
                f"  Initial Processing: {self.processing_times.get('initial_processing', 0):.2f}s",
                f"  Compound Analysis: {self.processing_times.get('compound_analysis', 0):.2f}s",
                f"  Total Time: {sum(self.processing_times.values()):.2f}s"
            ])
            
        return "\n".join(lines)

class KeywordProcessor:
    """ Handles keyword processing pipeline """
    def __init__(self, container: KeywordContainer, config: Config):
        self.container = container
        self.config = config
        self.embeddings = None
        generation_params = {
            "max_context": 4096,
            "max_length": 100,
            "top_p": 0.95,
            "top_k": 40,
            "temp": 0.4,
            "rep_pen": 1.01,
            "min_p": 0.05,
        }
        self.core = KoboldAPICore(config.api_url, generation_params)
        
        # Color descriptions are annoying because it makes multiples of everything
        # that can have a color: "blue cup", "red cup", "white cup"
        self.colors = {
            'red', 'blue', 'green', 'yellow', 'purple', 'orange',
            'white', 'black', 'gray', 'grey', 'brown', 'beige',
            'pink', 'turquoise', 'golden', 'plaid', 'clear', 'metallic'
        }    
    
    def process(self):
        self.stats = ProcessingStats()
        self.process_times = {}
        
        if not self.container.has_keywords:
            start = time.time()
            
            # Extract initial keywords
            self.container.raw_keywords = self.extract_keywords(self.container.file_metadata)
            self.stats.initial_keywords = len(self.container.raw_keywords)
            
            # Process color terms
            self.container.raw_keywords = self._remove_color_prefixes(self.container.raw_keywords)
            self.container.keywords = list(set(self.container.raw_keywords))
            self.stats.unique_keywords = len(self.container.keywords)
            
            # Extract compounds
            self.container.raw_compounds = self.extract_compounds(self.container.raw_keywords)
            self.stats.compound_keywords = len(self.container.raw_compounds)
            self.container.compounds = self.extract_compounds(self.container.keywords)
            self.stats.unique_compounds = len(self.container.compounds)

            # Extract singles
            self.container.singles = self.extract_singles(self.container.raw_keywords)
            self.stats.single_keywords = len(self.container.singles)
            self.stats.unique_singles = len(set(self.container.singles))
            
            self.process_times['initial_processing'] = time.time() - start
            
            start = time.time()
            self._analyze_compound_structure()
            self.process_splits(self.container.raw_compounds)
            self.process_times['compound_analysis'] = time.time() - start
            
        if not self.config.skip_embeds and not self.container.has_candidates:
            print(f"Generating embeddings...")
            self.embeddings = self.generate_embeddings()
            #self.stats.embeddings_generated = len(self.embeddings)
            self.container.candidate_mappings = self.map_candidates()
            #self.stats.candidate_pairs = sum(len(candidates) 
            #    for candidates in self.container.candidate_mappings.values())
            
        if not self.config.skip_candidates and not self.container.has_expansions:
            self.container.keyword_expansions = self.validate_candidates()
            self.stats.verified_candidates = sum(len(expansions) 
                for expansions in self.container.keyword_expansions.values())
            self.stats.final_keywords = (len(self.container.keywords) + 
                self.stats.verified_candidates)
        
        self.stats.processing_times = self.process_times
            
    def pre_splits(self, compounds: List[str]):
        """ Analyze compounds for statistical patterns suggesting splits 
            
            Compound word = modifier + base
            Compounds with same starting modifier that are unique, divided
            by total compounds with the starting modifier approaches 1
            makes chance of modifier being a descriptor higher. If 
            modifier is a descriptor, we can split the compound.
        """
        
        modifier_totals = defaultdict(int)
        modifier_uniques = defaultdict(set)
        
        for compound in compounds:
            words = compound.split()
            if len(words) != 2:  # Skip compounds that aren't exactly two words
                continue
            modifier, base = words
            
            # Track counts and partnerships
            modifier_totals[modifier] += 1
            modifier_uniques[modifier].add(compound)

        to_split_modifier = set()

        for modifier in modifier_totals:
            total = modifier_totals[modifier]
            uniques = len(modifier_uniques[modifier])
            if total > 6:
                if (uniques / total) > 0.8: 
                    to_split_modifier.add(modifier)
        return to_split_modifier

    def extract_singles(self, keywords: List[str]) -> List[str]:
        """ Extract single-word keywords """
        return [k for k in keywords if len(k.split()) == 1]

    def process_splits(self, compounds: List[str]):
        """ Process all compound splitting methods """
        
        to_split_modifier = self.pre_splits(compounds)
        to_split = [] 
        
        for compound in compounds:
            if compound.split()[0] in to_split_modifier:
                to_split.append(compound)
                
        remaining = [c for c in compounds if c not in to_split]
        to_split = list(set(to_split))
        remaining = list(set(remaining))
        if to_split:
            splits = []
            for compound in to_split:
                self.stats.split_compounds += 1
                print(f"Splitting {self.stats.split_compounds} of {len(to_split)}: {compound}") 
                words = compound.split()
                if words[1] in ['and', 'or']:
                    splits.append(words[0])
                    splits.append(words[2])
                else:
                    splits.extend(words)
            # Remove split compounds
            self.container.keywords = [k for k in self.container.keywords 
                                     if k not in to_split]
            self.container.keywords.extend(splits)
            self.container.keywords = list(set(self.container.keywords))

    def _analyze_compound_structure(self):
        """ Analyze structure of compound keywords """
        modifiers = set()
        bases = set()
        
        for compound in self.container.compounds:
            words = compound.split()
            if len(words) == 2:
                modifiers.add(words[0])
                bases.add(words[1])
                
        self.stats.unique_modifiers = len(modifiers)
        self.stats.unique_bases = len(bases)
        
    def extract_keywords(self, entries) -> List[str]:
        """ Extract keywords from metadata """
        keywords = []        
        for entry in entries:
            if 'Composite:Keywords' in entry:
                keywords.extend(entry['Composite:Keywords'])
            elif 'Subject' in entry:
                keywords.extend(entry['Subject'])
            else:
                if isinstance(entry, list):
                    keywords.extend(entry)
        return keywords
    
    def extract_compounds(self, keyword_list) -> List[str]:
        """ Find multi-word keywords that are 2-3 words long. """
        compounds = []
        for entry in keyword_list:
            words = str(entry).split()
            if 1 < len(words) < 4:  # Keep compounds of 2-3 words
                compounds.append(entry)
        return compounds 
        
    def generate_embeddings(self) -> np.ndarray:
        """ Generate embeddings using llama.cpp """
        embeddings_dict = {'data': []}
        batch_size = 32
        batch_number = 0
        self.container.keywords = self.container.keywords
        self.stats.embeddings_generated = len(self.container.keywords)
        
        for i in range(0, len(self.container.keywords), batch_size):
            batch = self.container.keywords[i:i + batch_size]
            batch_number += batch_size
            
            #print(f"Embedded {batch_number} of {len(self.container.keywords)}: {', '.join(batch)}")
            cmd = [
                str(self.config.llama_path),
                "-m", str(self.config.model_path),
                "--embd-normalize", "2",
                "--embd-output-format", "json",
                #"--pooling", "mean",
                "-c", "512",
                "-p", '\n'.join(batch)
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                batch_embeddings = json.loads(result.stdout)
                embeddings_dict['data'].extend(batch_embeddings['data'])
                        
            except subprocess.CalledProcessError as e:
                print(f"Process error: {e}")
                print(f"Error output: {e.stderr}")
                continue
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                continue
                    
        if not embeddings_dict['data']:
            raise RuntimeError("Failed to generate any valid embeddings")

        embeddings_list = [item['embedding'] for item in embeddings_dict['data']]
        embeddings = np.array(embeddings_list, dtype='float32')
        faiss.normalize_L2(embeddings)
        
        return embeddings
      
    def map_candidates(self) -> Dict[str, List[str]]:
        """ Find candidate matches using FAISS """
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(self.embeddings)
        candidate_mapping = {}
        for idx, tag in enumerate(self.container.keywords):
            query_embedding = self.embeddings[idx].reshape(1, -1)
            distances, indices = index.search(
                query_embedding, 
                self.config.max_candidates + 1
            )        
            candidates = [
                self.container.keywords[i] 
                for i, dist in zip(indices[0], distances[0])
                if self.container.keywords[i] != tag 
                and dist < self.config.similarity_threshold
            ][:self.config.max_candidates]
            
            if candidates:
                filtered_candidates = set()
                
                for candidate in candidates:
                        
                    # Remove compounds and single letter keywords from candidates
                    if len(candidate.split()) != 2 and len(candidate) != 1:
                        filtered_candidates.add(candidate)
                    
                candidate_mapping[tag] = list(filtered_candidates)
                self.stats.candidate_pairs += len(filtered_candidates)    
        return candidate_mapping
        
    def validate_candidates(self) -> Dict[str, List[str]]:
        """ Validate candidates using LLM """
        
        prompt = """Your task is to identify exact synonyms for the input word and parents of the input word. 
IMPORTANT: The synonyms will always be ABOVE the input word in the parent hierarchy and thus a MORE GENERAL describer.

Find EXACT SYNONYMS in the list. Only use words included in the list; do not add any addition words.
Include ONLY if:
   - Words mean EXACTLY the same thing (like "car" = "automobile")
   - Can substitute in ANY context with NO change in meaning
   - Word is in the list
   
Find GENERAL TERMS or CATEGORIES that fit ABOVE the input word in the hierarchy list.
Include ONLY if:
   - Parent is MORE GENERAL than input word
   - Word is in the list
   
CRITICAL RELATIONSHIP DIRECTION:
VALID "metal" -> "material" (VALID: metal is a type of material)
INVALID "metal" -> "brass" (INVALID: brass is a type of metal - wrong direction!)
INVALID "metal" -> "bronze" (INVALID: bronze is a type of metal - wrong direction!)

EXAMPLE OUTPUTS:
Input: "metal"
Candidates: "brass, bronze, gold, material, substance"
Valid output: {"metal": ["material", "substance"]}  # only parent categories, NO subtypes

Input: "dog"
Candidates: "poodle, animal, pet, mammal, canine"
Valid output: {"dog": ["animal", "mammal", "canine"]}  # only parent categories and synonyms

Only use the words provided in the candidates list. Do NOT add any new words. You do not have to use all or any of the words in the list and can return an empty list.

Reply with a JSON object as follows: { str: [str, ...] } 
"""
        synonym_mapping = {}
        i = 1
        for tag, candidates in self.container.candidate_mappings.items():
            i += 1
            try:
                result = json.loads(rj(self.core.wrap_and_generate(
                    instruction=prompt,
                    content=f'\nWord: "{tag}"\nCandidates: {", ".join(candidates)}\n'))
                )
                if isinstance(result, dict):
                    result_list = result.get(tag, [])
                    
                    if result_list:
                        #check if the LLM made up candidates
                        result_list = self._remove_pretend_words(list(set(result_list)), candidates) 
                        synonym_mapping[tag] = result_list
                        print(f"Validated {i} of {len(self.container.keywords)} {tag}: {result_list}")
                        self.stats.verified_candidates += len(result_list)
            except Exception as e:
                print(f"Error validating {tag}: {str(e)}")
                continue
            
        return synonym_mapping

    def _remove_pretend_words(self, result_list, candidates):
        """ Check if the LLM made up any candidates and remove them """
        return [word for word in result_list if word in candidates]
        
    def _remove_color_prefixes(self, strings: List[str]) -> List[str]:
        """ Remove color descriptors from strings """
        result = []
        self.stats.color_compounds = 0
        
        for s in strings:
            words = str(s).split()
            if (len(words) == 2) and (words[0] in self.colors):
                result.append(words[1])
                self.stats.color_compounds += 1
            else:
                result.append(s)
                
        return result
        
def extract_entries_from_json(json_file: str):
    """ Loads and returns entries from a JSON file """
    with open(json_file, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    return entries
    
def save_json(data, path):
    """ Helper to save JSON files """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        print(f"Saved to {path}")
        
def main():
    parser = argparse.ArgumentParser(description='Semantic tag merger')
    parser.add_argument('--model-path', type=str,
                       default="all-MiniLM-L6-v2-ggml-model-f16.gguf",
                       help='Path to embedding model')
    parser.add_argument('--llama-path', type=str,
                       default="llama-embedding.exe",
                       help='Path to llama-embeddings')
    parser.add_argument('directory', type=str,
                       help="Directory containing the files")
    parser.add_argument("--api-url", default="http://localhost:5001",
                       help="URL for the LLM API")
    args = parser.parse_args()
    
    config = Config.from_args(args)
    
    if not os.path.exists(config.directory):
        print(f"Directory not found: {config.directory}")
        sys.exit(1)
    
    print(f"Extracting metadata from {config.directory}")
    if os.path.exists(os.path.join(config.directory, 'KeywordExpander_metadata.json')):
        metadata = extract_entries_from_json(os.path.join(config.directory, 'KeywordExpander_metadata.json'))
        
    else:
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_tags(config.directory, ["MWG:Keywords"], "-r")
        
        if metadata:
            path = os.path.join(config.directory, "KeywordExpander_metadata.json")
            save_json(metadata, path)
    
    container = KeywordContainer(metadata)
    processor = KeywordProcessor(container, config)
    
    # Run processing pipeline
    processor.process()
    print(processor.stats)
    
    if container.has_expansions:
        save_json(container.keyword_expansions, "KeywordExpander_expansions.json")
    
    # Write expanded keywords back to files
    if container.has_expansions:
        expanded_entries = container.prepare_output()
        print(f"Expanding metadata...")
        with exiftool.ExifToolHelper() as et:
            for entry in expanded_entries:
                if 'SourceFile' not in entry or 'Composite:Keywords' not in entry:
                    continue
                    
                file_path = Path(config.directory) / entry['SourceFile']
                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue
                    
                if not os.access(file_path, os.W_OK):
                    print(f"No write permission for file: {file_path}")
                    continue
                    
                metadata = {'MWG:Keywords': entry['Composite:Keywords']}
                
                try:
                    et.set_tags(
                        str(file_path),
                        tags=metadata,
                        params=["-P", "-overwrite_original"],
                    )
                    #print(f"{file_path}: Success!")
                except Exception as e:
                    print(f"Error updating metadata for {file_path}: {str(e)}")
    
    print("Processing complete!")
    
if __name__ == '__main__':
    main()
    