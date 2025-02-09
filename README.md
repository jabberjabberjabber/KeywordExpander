# Keyword Expander

A tool for processing and expanding image metadata tags using semantic similarity and LLM validation. This tool helps organize image collections by identifying and grouping related tags while maintaining semantic hierarchies.

## Features

- Extracts existing metadata tags from images
- Generates semantic embeddings for tags using llama.cpp
- Identifies semantically similar tags using FAISS
- Validates tag relationships using LLM
- Updates image metadata with expanded tag sets
- Preserves semantic hierarchies (e.g., "metal" -> "material" but not "metal" -> "brass")

## Requirements

- Python 3.8+
- ExifTool
- llama.cpp with embedding support
- Kobold API compatible LLM server

Python dependencies:
```
exiftool
numpy
faiss-cpu
json-repair
```

## Installation

1. Install ExifTool for your platform: https://exiftool.org/
2. Clone this repository
3. Install Python dependencies: `pip install -r requirements.txt`
4. Download a GGUF format embedding model (e.g., all-MiniLM-L6-v2) and language model (e.g., gemma-2-2b-it)
5. Build or obtain llama.cpp and place llamacpp-embedding and needed binaries in the KeywordExpander folder
6. Download Koboldcpp launch it with the language model
   

## Usage

Basic usage:
```bash
python postprocess.py /path/to/image/directory \
    --model-path /path/to/embedding-model.gguf \
    --llama-path /path/to/llama-embedding \
    --api-url http://localhost:5001
```

The script will:
1. Extract current metadata from images
2. Generate tag embeddings
3. Find similar tags
4. Validate relationships using LLM
5. Update image metadata with expanded tags

## Output Files

The script generates three JSON files in the target directory:
- `01_metadata.json`: Raw metadata from images
- `02_mapped_candidates.json`: Initial tag similarity mappings
- `03_mapped_keywords.json`: LLM-validated tag relationships

## Credits
Invaluable assistance provided by ocha221. After reaching out for solutions they kindly wrote a working implementation of the idea:
- https://github.com/ocha221/semantic-tagging-tools

I rewrote it to use local models and to integrate with LlamaImageTagger for keyword postprocessing. 

## License

This project is licensed under GPLv3. See the LICENSE file for details.
semantic-tagging-tools is licensed under MIT.
Llama.cpp is licensed under MIT.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
