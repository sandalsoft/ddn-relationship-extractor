# YAML Relationship Extractor

This script extracts specific YAML stanzas (`kind: Relationship`) from one or more input YAML files and moves them into a designated output file.

## Features

- Processes multiple input YAML files using glob patterns (e.g., `*.yaml`, `**/metadata/*.yml`).
- Identifies and extracts all YAML documents where the top-level `kind` key has the value `Relationship`.
- Moves extracted stanzas to a specified output file.
- Creates backups of original input files (`.backup_<filename>`) before modification.
- Verifies that all extracted stanzas are present in the output file before removing them from the input files.
- Includes error handling and logging.
- Removes backups only upon successful completion.

## Requirements

- Python 3.7+
- PyYAML (`pip install -r requirements.txt`)

## Usage

```bash
python main.py [INPUT_GLOB_PATTERN ...] -o OUTPUT_FILE
```

**Arguments:**

- `INPUT_GLOB_PATTERN`: One or more glob patterns matching the input YAML files. Shell expansion usually works, but quoting patterns might be necessary depending on your shell (e.g., `'*.yaml'` or `"**/*.yml"`).
- `-o OUTPUT_FILE`, `--output OUTPUT_FILE`: Path to the YAML file where extracted `Relationship` stanzas will be written. This file will be overwritten if it exists.

**Example:**

```bash
# Process all .yaml files in the current directory and its subdirectories
python main.py '**/*.yaml' -o extracted_relationships.yaml

# Process specific files
python main.py tables.yaml metadata/more_rels.yml -o all_relationships.yaml
```

## Development

Unit tests are included:

```bash
pip install -r requirements.txt
python -m unittest test_main.py
```
