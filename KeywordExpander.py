import json
import argparse
import exiftool
import os
import sys
import numpy as np
import faiss
import subprocess
from typing import List, Dict, Optional
from pathlib import Path
from koboldapi import KoboldAPICore
from json_repair import repair_json as rj

def extract_entries_from_json(json_file: str) -> List[Dict]:
    """ Loads and returns entries from a JSON file
    
        Args:
            json_file: Path to JSON file containing entries
            
        Returns:
            List of dictionary entries from the JSON file
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    return entries
    
def save_to_json(data: Dict, json_file: str) -> None:
    """ Saves data to a JSON file with pretty printing
    
        Args:
            data: Data to save
            json_file: Output file path
    """
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {json_file}")

def extract_keywords_from_entries(entries: List[Dict]) -> List[str]:
    """ Extracts keywords from the Keywords field of entries
    
        Args:
            entries: List of metadata entries
            
        Returns:
            Deduplicated list of keywords found in Keywords fields
    """
    if not isinstance(entries, list):
        print("Error: bad formatting. Supposed to be a list of dicts.")
        return []
    
    keywords = []    
    for entry in entries:
        if 'Composite:Keywords' in entry and isinstance(entry['Composite:Keywords'], list):
            keywords.extend(entry['Composite:Keywords'])
    
    return list(set(keywords))

def get_metadata(dir_path: str) -> List[Dict]:
    """ Extracts metadata from files in directory using exiftool
    
        Args:
            dir_path: Directory path to process
            
        Returns:
            List of metadata dictionaries
    """
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_tags(dir_path, ["MWG:Keywords"], "-r")
    return metadata
        
def remove_multiword_expansions(concept_lists: List[List]) -> List[List]:
    """ Takes a list of concept expansions and removes any expansion terms 
        that contain multiple words.
        
        Args:
            concept_lists: List of [concept, [expansions]] pairs where
                concept is a string and expansions is a list of strings
        
        Returns:
            Same structure as input but with multi-word expansions removed
    """
    result = []
    
    for keywords, expansions in concept_lists:
        filtered_expansions = []
        for term in expansions:
            term = str(term)
            if len(term.split()) == 1:
                filtered_expansions.append(term)
       
        result.append([keywords, filtered_expansions])
        
    return result 
    
def combine_keywords_candidates(
    keyword_groups: Dict[str, List[str]], 
    candidate_groups: List[Dict]
) -> List[Dict]:
    """ Takes a list of dicts with optional Keywords keys and a dict of keyword 
        groups and combines matching entries.
        
        Only includes entries that have a Keywords key in the result.
        
        Args:
            keyword_groups: Dict mapping keywords to lists of values
            candidate_groups: List of dicts with SourceFile and optional Keywords
                
        Returns:
            List of dicts with combined Keywords lists
    """
    result = []
    
    for entry in candidate_groups:
        if 'Composite:Keywords' not in entry:
            continue
            
        new_entry = entry.copy()
        candidate_set = set(entry['Composite:Keywords'])
        for keyword in keyword_groups:
            if keyword in candidate_set:
                candidate_set.update(keyword_groups[keyword])
                new_entry['Composite:Keywords'] = list(candidate_set)
                
        result.append(new_entry)
    return result

def write_keywords(combined_entries: List[Dict], base_path: str) -> None:
    """ Updates XMP metadata for a collection of file entries
    
        Args:
            combined_entries: List of dicts containing file metadata
            base_path: Base directory path for resolving relative paths
    """
    with exiftool.ExifToolHelper() as et:
        for entry in combined_entries:
            if 'SourceFile' not in entry or 'Composite:Keywords' not in entry:
                continue
                
            file_path = Path(base_path) / entry['SourceFile']
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
                print(f"{file_path}: Success!")
            except Exception as e:
                print(f"Error updating metadata for {file_path}: {str(e)}")

def generate_embeddings(
    model_path: str, 
    llama_path: str, 
    tags: List[str]
) -> np.ndarray:
    """ Generates embeddings for a list of tags using llama.cpp
        
        Args:
            model_path: Path to the embedding model
            llama_path: Path to llama.cpp executable
            tags: List of strings to generate embeddings for
            
        Returns:
            Normalized embeddings matrix
            
        Raises:
            RuntimeError: If no valid embeddings could be generated
    """
    embeddings_dict = {'data': []}
    batch_size = 1
    batch_number = 0
    
    for i in range(0, len(tags), batch_size):
        batch = tags[i:i + batch_size]
        batch_number += 1
        
        print(f"Batch {batch_number}: {batch[0]}")
        cmd = [
            str(llama_path),
            "-m", str(model_path),
            "--embd-normalize", "2",
            "--embd-output-format", "json",
            "--pooling", "mean",
            "-c", "512",
            "-p", batch[0]
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Raw output: {result.stdout[:20]}...")
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

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """ Build FAISS index for similarity search 
    
        Args:
            embeddings: Matrix of embeddings to index
            
        Returns:
            FAISS L2 index containing the embeddings
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def remove_color_prefixed_strings(string_list: List[str]) -> List[str]:
    """ Takes a list of strings and removes entries starting with common colors.
        Case insensitive matching is used.
        
        Args:
            string_list: List of strings to filter
            
        Returns:
            Filtered list with color-prefixed strings removed
    """
    colors = {
        'red', 'blue', 'green', 'yellow', 'purple', 'orange',
        'white', 'black', 'gray', 'grey', 'brown', 'beige',
        'pink', 'navy', 'maroon', 'crimson', 'turquoise'
    }
    
    filtered_list = []
    for string in string_list:
        string = str(string)
        words = string.strip().split()
        if not words or words[0].lower() not in colors:
            filtered_list.append(string)
            
    return filtered_list
    
def process_entries(api_url: str, entries: List[List]) -> Dict[str, List[str]]:
    """ Process entries through LLM API to validate synonym relationships
    
        Args:
            api_url: URL for the Kobold API endpoint
            entries: List of [tag, candidates] pairs to process
            
        Returns:
            Dictionary mapping tags to their validated synonyms
    """
    core = KoboldAPICore(api_url)
    synonym_mapping = {}
    
    for subject, candidates in entries:
        try:
            synonyms = get_synonyms(core, subject, candidates)
            if synonyms:
                synonym_mapping[subject] = synonyms
        except Exception as e:
            print(f"Error processing entry {subject}: {str(e)}")
            continue
        
    return synonym_mapping  

def get_synonyms(core: KoboldAPICore, tag: str, candidates: List[str]) -> List[str]:
    """ Get validated synonyms using KoboldAPI 
    
        Args:
            core: KoboldAPI instance for LLM interaction
            tag: Target word to find synonyms for
            candidates: List of potential synonym candidates
            
        Returns:
            List of validated synonyms for the tag
    """
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

AUTOMATIC EXCLUSION:
- Any subtypes or specific varieties of the input word
- Related terms that aren't strictly synonyms
- Specific examples of the input category

Input: "dog"
Candidates: "poodle, animal, pet, mammal, canine"
Valid output: {"dog": ["animal", "mammal"]}  # only parent categories

Only use the words provided in the candidates list. Do NOT add any new words. You do not have to use all or any of the words in the list and can return an empty list.
Reply with EXACTLY: { str: [str, ...] } 
"""
    try:       
        result = json.loads(rj(core.wrap_and_generate(
            instruction=prompt,
            content=f'\nWord: "{tag}"\nCandidates: {", ".join(candidates)}\n'))
        )
        if isinstance(result, dict):
            result_list = result.get(tag, [])
            return list(result_list)
        return []
    except Exception as e:
        print(f"Error getting synonyms for {tag}: {str(e)}")
        return []
        
def map_candidates(tags: List[str], 
    index: faiss.IndexFlatL2, 
    embeddings: np.ndarray,
    similarity_threshold: float = 1.0,
    max_candidates: int = 10
) -> List[List]:
    """ Process all tags to generate synonym mapping 
    
        Args:
            tags: List of strings to process
            index: FAISS index for similarity search
            embeddings: Matrix of embeddings corresponding to tags
            similarity_threshold: Maximum distance for considering candidates
            max_candidates: Maximum number of candidates per tag
            
        Returns:
            List mapping tags to their synonyms
            [["keyword", ["candidate"]],...]
    """
    print("Processing tags...")
    print(f"Tags: {len(tags)}, Embeddings: {embeddings.shape[0]}")
    
    candidate_mapping = []        
    for idx, tag in enumerate(tags):
        query_embedding = embeddings[idx].reshape(1, -1)
        distances, indices = index.search(query_embedding, max_candidates + 1)
        candidates = [
            tags[i] for i, dist in zip(indices[0], distances[0])
            if tags[i] != tag and dist < similarity_threshold
        ][:max_candidates]
        
        if candidates:
            candidate_mapping.append([tag, candidates])
                
    return candidate_mapping    

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
    
    if not os.path.exists(args.directory):
        print(f"Directory not found: {args.directory}")
        sys.exit(1)
    
    # First, extract the metadata from the images
    print(f"Extracting metadata from {args.directory}")
    metadata = get_metadata(args.directory)
    if metadata:
        save_to_json(
            metadata, 
            os.path.join(args.directory, "01_metadata.json")
        )
    
    # Make a deduplicated list of keywords
    print(f"Extracting keywords...")
    keywords = extract_keywords_from_entries(metadata)
    
    # Remove some annoying, hard to correlate terms
    print(f"Removing color prefixed keywords...")
    keywords = remove_color_prefixed_strings(keywords)
    
    # Get the embeddings
    print(f"Generating embeddings...")
    embeddings = generate_embeddings(args.model_path, args.llama_path, keywords)
    
    # Create index 
    print(f"Building index...")
    index = build_index(embeddings)
    
    # Map the candidates
    print(f"Mapping candidates...")
    candidate_mapping = map_candidates(keywords, index, embeddings)
    
    # Remove multiple word keywords in candidates list
    print("Removing multi-word candidates...")
    candidate_mapping = remove_multiword_expansions(candidate_mapping)
    
    # Save candidates in json
    print("Saving candidates to json...")
    save_to_json(
        candidate_mapping, 
        os.path.join(args.directory, "02_mapped_candidates.json")
    )
    
    # Send the mappings to the LLM to check
    print("Matching candidate maps with LLM...")
    synonym_mapping = process_entries(args.api_url, candidate_mapping)
    
    # Save to json
    print("Saving mappings to json...")
    save_to_json(
        synonym_mapping,
        os.path.join(args.directory, "03_mapped_keywords.json")
    )
    
    # Combine the metadata with the new mappings
    print("Combining metadata with mappings...")
    combined_entries = combine_keywords_candidates(synonym_mapping, metadata)
    
    # Write the metadata
    print("Writing expanded metadata to image files...")
    write_keywords(combined_entries, args.directory)
    
    print("Processing complete!")

if __name__ == '__main__':
    main()
    