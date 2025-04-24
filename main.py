#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

import argparse
import glob
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract YAML stanzas of kind: Relationship and move them to a separate file or print to stdout.')
    parser.add_argument('input_patterns', nargs='+',
                        help='Glob pattern(s) for input YAML files (e.g., "*.yaml", "**/data.yml").')
    parser.add_argument('-o', '--output', required=False, default=None,
                        help='Path to the output YAML file. If omitted, output is written to stdout and input files are not modified.')
    parser.add_argument('-r', '--remove-from-source', action='store_true',
                        help='Remove the extracted Relationship stanzas from the original input files. Requires confirmation.')
    return parser.parse_args()


def find_files(patterns: List[str]) -> List[Path]:
    """Finds files matching the given glob patterns."""
    files = set()
    for pattern in patterns:
        # Expanduser to handle ~ and enable recursive globbing with **
        expanded_pattern = os.path.expanduser(pattern)
        found_files = glob.glob(expanded_pattern, recursive=True)
        if not found_files:
            logging.warning(f"No files found matching pattern: {pattern}")
        for f in found_files:
            path = Path(f).resolve()  # Get absolute path
            if path.is_file():
                files.add(path)
            else:
                logging.warning(
                    f"Pattern matched a directory, skipping: {path}")

    logging.info(f"Found {len(files)} input files to process.")
    if not files:
        logging.error("No valid input files found. Exiting.")
        sys.exit(1)
    return sorted(list(files))  # Return a sorted list for consistent order


def create_backups(files: List[Path]) -> List[Path]:
    """Creates backup copies of the input files."""
    backup_files = []
    logging.info("Creating backups...")
    for file_path in files:
        backup_path = file_path.parent / f".backup_{file_path.name}"
        try:
            shutil.copy2(file_path, backup_path)  # copy2 preserves metadata
            backup_files.append(backup_path)
            logging.info(f"Created backup: {backup_path}")
        except Exception as e:
            logging.error(f"Failed to create backup for {file_path}: {e}")
            # Attempt cleanup of any backups created so far
            remove_backups(backup_files)
            sys.exit(1)
    return backup_files


def remove_backups(backup_files: List[Path]):
    """Removes the backup files."""
    logging.info("Removing backups...")
    for backup_path in backup_files:
        try:
            os.remove(backup_path)
            logging.info(f"Removed backup: {backup_path}")
        except OSError as e:
            logging.warning(f"Failed to remove backup file {backup_path}: {e}")


def read_yaml_documents(file_path: Path) -> Iterable[Dict[str, Any]]:
    """Reads a YAML file and yields each document."""
    try:
        with open(file_path, 'r') as f:
            # Use safe_load_all for multi-document files
            yield from yaml.safe_load_all(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {file_path}: {e}")
        # Decide how to handle parse errors, e.g., skip file or exit
        raise  # Re-raise to handle upstream
    except IOError as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise  # Re-raise to handle upstream


def is_relationship_stanza(doc: Any) -> bool:
    """Checks if a YAML document is a 'Relationship' stanza."""
    return isinstance(doc, dict) and doc.get('kind') == 'Relationship'


def extract_relationships(files: List[Path]) -> Tuple[List[Dict[str, Any]], Dict[Path, List[Dict[str, Any]]]]:
    """Extracts all 'Relationship' stanzas from the input files."""
    all_relationships = []
    relationships_by_file = {file_path: [] for file_path in files}
    logging.info("Extracting 'Relationship' stanzas...")

    for file_path in files:
        logging.debug(f"Processing file: {file_path}")
        try:
            doc_index = 0
            for doc in read_yaml_documents(file_path):
                if doc is None:  # Handle empty documents (e.g. just '---')
                    doc_index += 1
                    continue
                # Store original index for potential later use if needed
                # doc['_original_index'] = doc_index
                # doc['_source_file'] = str(file_path) # Add metadata before check
                if is_relationship_stanza(doc):
                    logging.debug(f"Found Relationship stanza in {file_path}")
                    # Make a copy to avoid modifying original dicts if needed elsewhere
                    relationship_copy = doc.copy()
                    all_relationships.append(relationship_copy)
                    relationships_by_file[file_path].append(relationship_copy)
                doc_index += 1
        except Exception as e:
            # Error handled in read_yaml_documents, but catch here for safety
            logging.error(
                f"Skipping file {file_path} due to previous error: {e}")
            # Potentially remove file from further processing or halt
            continue  # Continue with next file for now

    logging.info(
        f"Extracted {len(all_relationships)} 'Relationship' stanzas in total.")
    return all_relationships, relationships_by_file


def write_output_file(output_path: Path, relationships: List[Dict[str, Any]]):
    """Writes the extracted relationships to the output file."""
    logging.info(f"Writing {len(relationships)} stanzas to {output_path}...")
    try:
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump_all(relationships, f,
                          default_flow_style=False, sort_keys=False)
        logging.info("Successfully wrote output file.")
    except IOError as e:
        logging.error(f"Failed to write output file {output_path}: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(
            f"Failed to serialize YAML for output file {output_path}: {e}")
        sys.exit(1)


def write_to_stream(stream: Any, relationships: List[Dict[str, Any]]):
    """Writes the extracted relationships to the given stream (e.g., stdout)."""
    logging.info(
        f"Writing {len(relationships)} stanzas to the output stream...")
    try:
        yaml.dump_all(relationships, stream,
                      default_flow_style=False, sort_keys=False)
        # Add a newline at the end for cleaner terminal output
        stream.write("\n")
        logging.info("Successfully wrote to output stream.")
    except Exception as e:  # Catch broader exceptions for stream writing
        logging.error(f"Failed to write to output stream: {e}")
        sys.exit(1)


def verify_and_remove(
    output_path: Path,
    original_relationships_by_file: Dict[Path, List[Dict[str, Any]]],
    all_extracted_relationships: List[Dict[str, Any]]
) -> bool:
    """Verifies relationships exist in output and removes them from input files."""
    logging.info("Verifying output and removing stanzas from input files...")

    try:
        output_relationships = list(read_yaml_documents(output_path))
    except Exception as e:
        logging.error(
            f"Failed to read back output file {output_path} for verification: {e}")
        return False

    # --- Verification Step 1: Count ---
    if len(output_relationships) != len(all_extracted_relationships):
        logging.error(
            f"Verification failed: Expected {len(all_extracted_relationships)} stanzas in output, found {len(output_relationships)}.")
        return False
    logging.info(
        "Verification: Output file stanza count matches extracted count.")

    # --- Verification Step 2: Content (more robust check) ---
    # Convert to a comparable format (e.g., frozenset of items for dicts) to handle order differences
    def make_hashable(obj: Any):
        """Recursively converts mutable objects (dict, list) into hashable types (tuple)."""
        if isinstance(obj, dict):
            # Sort items by key to ensure consistent hashing regardless of original order
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        if isinstance(obj, list):
            return tuple(make_hashable(elem) for elem in obj)
        # Assume other types (str, int, float, bool, None, tuple, frozenset) are already hashable
        # If other unhashable types exist in your YAML, they might need specific handling
        return obj

    try:
        output_set = {make_hashable(doc)
                      for doc in output_relationships if doc}
        extracted_set = {make_hashable(doc)
                         for doc in all_extracted_relationships if doc}
    except TypeError as e:
        logging.error(
            f"Verification failed: Could not create comparable sets from YAML documents (potential unhashable types): {e}")
        return False

    if output_set != extracted_set:
        logging.error(
            "Verification failed: Content mismatch between extracted relationships and output file content.")
        # Log differences for debugging if needed
        missing_in_output = extracted_set - output_set
        extra_in_output = output_set - extracted_set
        if missing_in_output:
            logging.error(f"Stanzas missing in output: {missing_in_output}")
        if extra_in_output:
            logging.error(f"Extra stanzas found in output: {extra_in_output}")
        return False
    logging.info(
        "Verification: Output file content matches extracted stanzas.")

    # --- Removal Step ---
    total_removed_count = 0
    # Keep order for removal logic if needed
    output_hashable_list = [make_hashable(doc)
                            for doc in output_relationships if doc]

    for file_path, original_relationships in original_relationships_by_file.items():
        if not original_relationships:  # Skip files that had no relationships
            continue

        logging.debug(f"Processing removals for file: {file_path}")
        stanzas_to_keep = []
        removed_count_for_file = 0
        try:
            needs_rewrite = False
            for doc in read_yaml_documents(file_path):
                if doc is None:
                    # Preserve separators/empty docs
                    stanzas_to_keep.append(doc)
                    continue

                doc_hashable = make_hashable(doc)
                # Check if it's one of the relationships *originally* from *this* file AND present in the output
                is_target_relationship = is_relationship_stanza(
                    doc) and doc_hashable in output_set

                # Double check it was meant to be extracted from this specific file originally
                # (This check is somewhat redundant given the `original_relationships_by_file` structure
                # and the set comparison, but adds safety)
                # originally_from_this_file = any(make_hashable(orig_doc) == doc_hashable for orig_doc in original_relationships)

                if is_target_relationship:  # and originally_from_this_file: # Redundant check
                    logging.debug(f"Removing stanza from {file_path}: {doc}")
                    removed_count_for_file += 1
                    needs_rewrite = True
                    # Do not add to stanzas_to_keep
                else:
                    stanzas_to_keep.append(doc)

            if needs_rewrite:
                logging.info(
                    f"Rewriting {file_path} after removing {removed_count_for_file} stanza(s).")
                try:
                    with open(file_path, 'w') as f:
                        # Use safe_dump_all, handle None for separators
                        if not stanzas_to_keep:  # If file becomes empty
                            f.write("")  # Write empty string
                        else:
                            # Need to handle potential None values if they represent '---'
                            # PyYAML's dump_all might handle this correctly, test needed
                            yaml.safe_dump_all([s for s in stanzas_to_keep if s is not None], f,
                                               default_flow_style=False, sort_keys=False)  # Basic filtering of None
                            # A more robust way might involve tracking '---' explicitly if PyYAML fails

                except (IOError, yaml.YAMLError) as e:
                    logging.error(
                        f"Failed to rewrite input file {file_path} after removal: {e}")
                    # CRITICAL: If rewrite fails, data consistency is compromised.
                    # Consider restoring backups or halting immediately.
                    return False  # Indicate failure
            else:
                logging.debug(
                    f"No stanzas removed from {file_path}, file not rewritten.")

            total_removed_count += removed_count_for_file

        except Exception as e:
            logging.error(
                f"Error during removal processing for {file_path}: {e}")
            return False  # Indicate failure

    # --- Final Check ---
    if total_removed_count != len(all_extracted_relationships):
        logging.error(
            f"Verification failed: Removed {total_removed_count} stanzas, but expected to remove {len(all_extracted_relationships)}.")
        return False

    logging.info(
        f"Successfully verified and removed {total_removed_count} stanzas from input files.")
    return True


def main():
    """Main execution function."""
    args = parse_arguments()
    stdout_mode = args.output is None
    remove_mode = args.remove_from_source and not stdout_mode

    # --- Early Exit Check for Removal Mode ---
    if remove_mode:
        logging.warning("The --remove-from-source flag is specified.")
        logging.warning(
            "This will MODIFY the original input files after verification.")
        user_confirmation = input(
            "Are you sure you want to proceed with modifying source files? (yes/no): ")
        if user_confirmation.lower() not in ['yes', 'y']:
            logging.info(
                "Operation aborted by user. No files will be processed or modified.")
            sys.exit(0)
        logging.info("Confirmation received. Proceeding...")
    elif args.remove_from_source and stdout_mode:
        # Inform user if -r is used with stdout mode (it will be ignored later)
        logging.warning(
            "Ignoring --remove-from-source flag as output is stdout. No source files will be modified.")

    input_files = find_files(args.input_patterns)
    if not input_files:
        return  # Error already logged by find_files

    if stdout_mode:
        logging.info("Outputting to stdout. Input files will not be modified.")
        # The warning for using -r with stdout is now handled earlier
        # if args.remove_from_source:
        #     logging.warning(
        #         "Ignoring --remove-from-source flag as output is stdout.")
        try:
            # Extract only, no backups or removal needed for stdout mode
            all_relationships, _ = extract_relationships(input_files)

            if not all_relationships:
                logging.info(
                    "No 'kind: Relationship' stanzas found. Nothing to output.")
                return

            write_to_stream(sys.stdout, all_relationships)
            logging.info("Extraction to stdout completed successfully.")

        except Exception as e:
            logging.error(
                f"An error occurred during extraction to stdout: {e}", exc_info=True)
            sys.exit(1)

    else:  # Outputting to a file
        output_path = Path(args.output).resolve()
        logging.info(f"Outputting to file: {output_path}.")

        if output_path.exists():
            logging.warning(
                f"Output file {output_path} already exists. It will be overwritten.")

        backup_files = []
        try:
            # --- Extract Relationships ---
            all_relationships, relationships_by_file = extract_relationships(
                input_files)

            if not all_relationships:
                logging.info(
                    "No 'kind: Relationship' stanzas found in any input file. Nothing to write to output file.")
                # Create/overwrite empty file
                write_output_file(output_path, [])
                return

            # --- Write to Output File ---
            write_output_file(output_path, all_relationships)

            # --- Removal Logic (only if remove_mode is true and confirmed) ---
            if remove_mode:
                # Confirmation already happened at the start
                logging.info(
                    "Proceeding with removal from source files...")
                backup_files = create_backups(input_files)

                if verify_and_remove(output_path, relationships_by_file, all_relationships):
                    logging.info(
                        "Process completed successfully. Removing backups.")
                    remove_backups(backup_files)
                else:
                    logging.error(
                        "Process failed during verification or removal. Backups WERE NOT removed.")
                    logging.error(
                        f"Input files may be in an inconsistent state. Backups are in: {', '.join(map(str, backup_files))}")
                    sys.exit(1)
            else:
                # This case covers when outputting to file but NOT removing
                logging.info(
                    "Extracted relationships written to output file. Source files were not modified.")

        except Exception as e:
            logging.error(
                f"An unexpected error occurred during file processing: {e}", exc_info=True)
            if remove_mode and backup_files:  # Check if backups were potentially created
                logging.error("Attempting to clean up backups due to error...")
                logging.error(
                    f"Backups were NOT removed due to error. They are located at: {', '.join(map(str, backup_files))}")
            sys.exit(1)


if __name__ == "__main__":
    main()
