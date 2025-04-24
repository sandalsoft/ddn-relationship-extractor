# YAML Relationship Extractor

This script extracts specific YAML stanzas (`kind: Relationship`) from one or more input YAML files and moves them into a designated output file.

## Features

- Processes multiple input YAML files using glob patterns (e.g., `*.yaml`, `**/metadata/*.yml`).
- Identifies and extracts all YAML documents where the top-level `kind` key has the value `Relationship`.
- Writes extracted stanzas to a specified output file or prints them to stdout.
- By default, source files are **not** modified.
- Optionally, removes extracted stanzas from the original source files if the `--remove-from-source` flag is used (requires confirmation).
- Creates backups of original input files (`.backup_<filename>`) before modification **only** when `--remove-from-source` is active.
- Verifies that all extracted stanzas are present in the output file before removing them from the input files (when removal is active).
- Includes error handling and logging.
- Removes backups only upon successful completion.

## Requirements

- Python 3.7+
- PyYAML (`pip install -r requirements.txt`)

## Usage

```bash
python main.py [INPUT_GLOB_PATTERN ...] [-o OUTPUT_FILE] [-r]
```

**Arguments:**

- `INPUT_GLOB_PATTERN`: One or more glob patterns matching the input YAML files. Shell expansion usually works, but quoting patterns might be necessary depending on your shell (e.g., `\'*.yaml\'` or `\"**/*.yml\"`).
- `-o OUTPUT_FILE`, `--output OUTPUT_FILE` (**Optional**): Path to the YAML file where extracted `Relationship` stanzas will be written.
  - If omitted, the extracted relationships will be printed to standard output (stdout). In this mode, input files are **not** modified, and no backups are created. The `-r` flag is ignored in stdout mode.
- `-r`, `--remove-from-source` (**Optional**): If specified **and** an output file (`-o`) is provided, the script will prompt for confirmation and then attempt to remove the extracted `Relationship` stanzas from the original input files after successfully writing them to the output file. Backups are created during this process. If this flag is not present, input files are never modified.

**Examples:**

```bash
# Example 1: Extract to a file (Default: does NOT modify input files)
# Process all .yaml files, write relationships to extracted_relationships.yaml
python main.py '**/*.yaml' -o extracted_relationships.yaml

# Example 2: Extract to a file AND remove from source (modifies input files after confirmation)
# Process all .yaml files, write relationships to extracted_relationships.yaml,
# and remove them from the original files.
python main.py '**/*.yaml' -o extracted_relationships.yaml -r

# Example 3: Output to stdout (does NOT modify input files)
# Process specific files and print relationships to the console
python main.py tables.yaml metadata/more_rels.yml

# Example 4: Pipe stdout to another command (does NOT modify input files)
# -r flag would be ignored here
python main.py config/**/*.yaml | grep 'name: user_posts'
```

## Development

Unit tests are included:

```bash
pip install -r requirements.txt
python -m unittest test_main.py
```
