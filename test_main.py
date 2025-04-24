import main
import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import argparse
import logging
import sys
from pathlib import Path
import io
import yaml

# Add the script directory to sys.path to import main
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import functions from main script
# We need to patch BEFORE importing if mocks need to affect module-level stuff
# or if imports trigger code execution we need to mock (not the case here)

# Disable logging during tests unless debugging
# logging.disable(logging.CRITICAL) # Keep logs enabled for debugging tests for now


class TestYamlRelationshipExtractor(unittest.TestCase):

    @patch('main.argparse.ArgumentParser')
    def test_parse_arguments(self, mock_parser_class):
        """Test argument parsing."""
        # Arrange
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = argparse.Namespace(
            input_patterns=['input/*.yaml', 'another/dir/**/*.yml'],
            output='output/relationships.yaml',
            remove_from_source=False  # Add default value
        )
        mock_parser_class.return_value = mock_parser_instance

        # Act
        args = main.parse_arguments()

        # Assert
        # Check standard args
        self.assertEqual(args.input_patterns, [
                         'input/*.yaml', 'another/dir/**/*.yml'])
        self.assertEqual(args.output, 'output/relationships.yaml')
        # Check calls to add_argument
        mock_parser_instance.add_argument.assert_any_call(
            'input_patterns', nargs='+', help=unittest.mock.ANY)
        # Fix: output is not required
        mock_parser_instance.add_argument.assert_any_call(
            '-o', '--output', required=False, default=None, help=unittest.mock.ANY)
        # Add check for remove_from_source argument
        mock_parser_instance.add_argument.assert_any_call(
            '-r', '--remove-from-source', action='store_true', help=unittest.mock.ANY)
        mock_parser_instance.parse_args.assert_called_once()

    @patch('main.glob.glob')
    @patch('main.Path.is_file')
    # Mock resolve to control path generation
    # @patch('main.Path.resolve') # Patching Path constructor below is often easier
    # Simple pass-through for expanduser
    @patch('os.path.expanduser', side_effect=lambda x: x)
    def test_find_files(self, mock_expanduser, mock_is_file, mock_glob):
        """Test finding files with glob patterns."""
        # Arrange
        # Simulate glob matching different patterns
        def glob_side_effect(pattern, recursive=False):
            if pattern == '*.yaml':
                return ['a.yaml', 'b.yaml']
            elif pattern == 'data/**/c.yaml':
                return ['data/sub/c.yaml', 'data/c.yaml']  # Test recursion
            elif pattern == 'nonexistent.yaml':
                return []
            elif pattern == 'a_dir':  # Test skipping directories
                return ['a_dir']
            else:
                return []
        mock_glob.side_effect = glob_side_effect

        # Simulate Path object behavior using Mocks that support comparison
        def create_mock_path(name, full_path_str, is_file):
            mock = MagicMock(spec=Path)
            mock.name = name
            mock.__str__ = MagicMock(return_value=full_path_str)
            mock.is_file = MagicMock(return_value=is_file)
            # Fix: Make mock comparable for sorting within main.find_files
            mock.__lt__ = lambda self, other: str(self) < str(other)
            # Mock resolve to return itself (simplified)
            mock.resolve = MagicMock(return_value=mock)
            return mock

        path_a = create_mock_path('a.yaml', '/abs/path/a.yaml', True)
        path_b = create_mock_path('b.yaml', '/abs/path/b.yaml', True)
        path_c1 = create_mock_path(
            'c.yaml', '/abs/path/data/sub/c.yaml', True)
        path_c2 = create_mock_path('c.yaml', '/abs/path/data/c.yaml', True)
        path_dir = create_mock_path('a_dir', '/abs/path/a_dir', False)

        # Map path strings (from glob) to our mock Path objects
        path_init_map = {
            'a.yaml': path_a,
            'b.yaml': path_b,
            'data/sub/c.yaml': path_c1,
            'data/c.yaml': path_c2,
            'a_dir': path_dir
        }

        # Patch Path constructor to return our pre-configured mocks
        with patch('main.Path', side_effect=lambda f: path_init_map[f]) as mock_path_constructor:
            # Act
            found_files = main.find_files(
                ['*.yaml', 'data/**/c.yaml', 'nonexistent.yaml', 'a_dir'])

            # Assert
            # Check glob calls
            mock_glob.assert_any_call('*.yaml', recursive=True)
            mock_glob.assert_any_call('data/**/c.yaml', recursive=True)
            mock_glob.assert_any_call('nonexistent.yaml', recursive=True)
            mock_glob.assert_any_call('a_dir', recursive=True)

            # Check Path instantiation and resolve calls
            self.assertEqual(mock_path_constructor.call_count, 5)
            # Resolve is called within Path(f).resolve()
            self.assertEqual(path_a.resolve.call_count, 1)
            self.assertEqual(path_b.resolve.call_count, 1)
            self.assertEqual(path_c1.resolve.call_count, 1)
            self.assertEqual(path_c2.resolve.call_count, 1)
            self.assertEqual(path_dir.resolve.call_count, 1)

            # Check is_file calls (called on the resolved path mock)
            self.assertEqual(path_a.is_file.call_count, 1)
            self.assertEqual(path_b.is_file.call_count, 1)
            self.assertEqual(path_c1.is_file.call_count, 1)
            self.assertEqual(path_c2.is_file.call_count, 1)
            self.assertEqual(path_dir.is_file.call_count, 1)

            # Check final list (should be sorted by Path's sorting, which we mocked using __lt__)
            expected_files = sorted(
                # Sort expected using str
                [path_a, path_b, path_c1, path_c2], key=str)
            # Direct comparison should work now
            self.assertEqual(found_files, expected_files)

    @patch('main.shutil.copy2')
    @patch('main.remove_backups')  # Mock remove_backups to check calls
    def test_create_backups(self, mock_remove_backups, mock_copy):
        """Test backup creation."""
        # Arrange
        file1_path = Path('/path/to/file1.yaml')
        file2_path = Path('/other/file2.yml')
        input_files = [file1_path, file2_path]

        backup1_path = file1_path.parent / f".backup_{file1_path.name}"
        backup2_path = file2_path.parent / f".backup_{file2_path.name}"

        # Act
        backup_files = main.create_backups(input_files)

        # Assert
        self.assertEqual(backup_files, [backup1_path, backup2_path])
        mock_copy.assert_has_calls([
            call(file1_path, backup1_path),
            call(file2_path, backup2_path)
        ])
        mock_remove_backups.assert_not_called()

    @patch('main.shutil.copy2', side_effect=IOError("Disk full"))
    @patch('main.remove_backups')
    @patch('sys.exit')  # Prevent test suite from exiting
    def test_create_backups_failure(self, mock_exit, mock_remove_backups, mock_copy):
        """Test backup creation failure and cleanup."""
        # Arrange
        file1_path = Path('/path/to/file1.yaml')
        input_files = [file1_path]
        backup1_path = file1_path.parent / f".backup_{file1_path.name}"

        # Act & Assert
        main.create_backups(input_files)

        # Assertions remain the same
        mock_copy.assert_called_once_with(file1_path, backup1_path)
        # Check that cleanup was attempted (even if no backups were successfully created yet)
        mock_remove_backups.assert_called_once_with([])
        mock_exit.assert_called_once_with(1)

    @patch('main.os.remove')
    def test_remove_backups(self, mock_remove):
        """Test backup removal."""
        # Arrange
        backup1_path = Path('/path/to/.backup_file1.yaml')
        backup2_path = Path('/other/.backup_file2.yml')
        backup_files = [backup1_path, backup2_path]

        # Act
        main.remove_backups(backup_files)

        # Assert
        mock_remove.assert_has_calls([
            call(backup1_path),
            call(backup2_path)
        ])

    @patch('main.os.remove', side_effect=OSError("Permission denied"))
    def test_remove_backups_failure(self, mock_remove):
        """Test backup removal failure (should log warning, not exit)."""
        # Arrange
        backup1_path = Path('/path/to/.backup_file1.yaml')
        backup_files = [backup1_path]

        # Act
        main.remove_backups(backup_files)

        # Assert
        mock_remove.assert_called_once_with(backup1_path)
        # No sys.exit expected

    @patch("builtins.open", new_callable=mock_open, read_data="""---
kind: Table
name: users
---
kind: Relationship
name: user_posts
---
kind: Config
value: true""")
    @patch('main.yaml.safe_load_all')
    def test_read_yaml_documents(self, mock_safe_load_all, mock_file):
        """Test reading multiple YAML documents from a file."""
        # Arrange
        file_path = Path('some/file.yaml')
        doc1 = {'kind': 'Table', 'name': 'users'}
        doc2 = {'kind': 'Relationship', 'name': 'user_posts'}
        doc3 = {'kind': 'Config', 'value': True}
        mock_safe_load_all.return_value = iter([doc1, doc2, doc3])

        # Act
        documents = list(main.read_yaml_documents(file_path))

        # Assert
        mock_file.assert_called_once_with(file_path, 'r')
        mock_safe_load_all.assert_called_once()
        self.assertEqual(documents, [doc1, doc2, doc3])

    @patch("builtins.open", new_callable=mock_open)
    @patch('main.yaml.safe_load_all', side_effect=yaml.YAMLError("Bad YAML"))
    def test_read_yaml_documents_parse_error(self, mock_safe_load_all, mock_file):
        """Test handling of YAML parsing errors."""
        # Arrange
        file_path = Path('bad/file.yaml')

        # Act & Assert
        with self.assertRaises(yaml.YAMLError):
            list(main.read_yaml_documents(file_path))  # Consume the generator
        mock_file.assert_called_once_with(file_path, 'r')
        mock_safe_load_all.assert_called_once()

    def test_is_relationship_stanza(self):
        """Test the relationship stanza check."""
        self.assertTrue(main.is_relationship_stanza(
            {'kind': 'Relationship', 'name': 'a'}))
        self.assertFalse(main.is_relationship_stanza(
            {'kind': 'Table', 'name': 'b'}))
        self.assertFalse(main.is_relationship_stanza(
            {'name': 'c'}))  # Missing kind
        self.assertFalse(main.is_relationship_stanza('not a dict'))
        self.assertFalse(main.is_relationship_stanza(None))
        self.assertFalse(main.is_relationship_stanza(
            [{'kind': 'Relationship'}]))

    @patch('main.read_yaml_documents')
    def test_extract_relationships(self, mock_read_yaml):
        """Test extracting relationship stanzas from multiple files."""
        # Arrange
        file1_path = Path('/path/file1.yaml')
        file2_path = Path('/path/file2.yaml')
        files = [file1_path, file2_path]

        doc1_1 = {'kind': 'Table', 'name': 't1'}
        doc1_2 = {'kind': 'Relationship', 'name': 'r1'}  # Target
        doc2_1 = {'kind': 'Relationship', 'name': 'r2'}  # Target
        doc2_2 = {'kind': 'Config', 'value': 1}
        doc2_3 = {'kind': 'Relationship', 'name': 'r3'}  # Target

        # Simulate return values for each file
        def read_side_effect(path):
            if path == file1_path:
                return iter([doc1_1, doc1_2])
            elif path == file2_path:
                return iter([doc2_1, doc2_2, doc2_3])
            else:
                return iter([])
        mock_read_yaml.side_effect = read_side_effect

        # Act
        all_rels, rels_by_file = main.extract_relationships(files)

        # Assert
        mock_read_yaml.assert_has_calls([call(file1_path), call(file2_path)])

        expected_all_rels = [doc1_2, doc2_1, doc2_3]
        expected_rels_by_file = {
            file1_path: [doc1_2],
            file2_path: [doc2_1, doc2_3]
        }

        # Simple comparison works as long as order is preserved and dicts are identical
        self.assertEqual(all_rels, expected_all_rels)
        self.assertEqual(rels_by_file, expected_rels_by_file)

    @patch('main.read_yaml_documents', side_effect=Exception("Read failed"))
    def test_extract_relationships_read_error(self, mock_read_yaml):
        """Test that extraction continues if one file fails to read/parse."""
        # Arrange
        file1_path = Path('/path/file1.yaml')  # This one will fail
        file2_path = Path('/path/file2.yaml')  # This one is ok
        files = [file1_path, file2_path]

        doc2_1 = {'kind': 'Relationship', 'name': 'r2'}

        def read_side_effect(path):
            if path == file1_path:
                raise Exception("Read failed")
            elif path == file2_path:
                return iter([doc2_1])
        mock_read_yaml.side_effect = read_side_effect

        # Act
        all_rels, rels_by_file = main.extract_relationships(files)

        # Assert
        mock_read_yaml.assert_has_calls([call(file1_path), call(file2_path)])
        self.assertEqual(all_rels, [doc2_1])  # Only relationship from file2
        # file1 has empty list
        self.assertEqual(rels_by_file, {file1_path: [], file2_path: [doc2_1]})

    @patch("builtins.open", new_callable=mock_open)  # Mock file opening
    @patch('main.yaml.dump_all')  # Mock YAML dumping
    @patch('main.Path.mkdir')  # Mock directory creation
    def test_write_output_file(self, mock_mkdir, mock_dump_all, mock_file):
        """Test writing relationships to the output file."""
        # Arrange
        output_path = MagicMock(spec=Path)
        output_path.parent = MagicMock(spec=Path)
        relationships = [
            {'kind': 'Relationship', 'name': 'r1'},
            {'kind': 'Relationship', 'name': 'r2'}
        ]

        # Act
        main.write_output_file(output_path, relationships)

        # Assert
        output_path.parent.mkdir.assert_called_once_with(
            parents=True, exist_ok=True)
        mock_file.assert_called_once_with(output_path, 'w')
        # Get the file handle passed to dump_all
        file_handle = mock_file()  # The return value of mock_open
        mock_dump_all.assert_called_once_with(
            relationships, file_handle, default_flow_style=False, sort_keys=False)

    # --- Tests for verify_and_remove --- More complex

    # Helper to create mock file content easily
    def create_yaml_content(self, *docs):
        stream = io.StringIO()
        yaml.dump_all(docs, stream, default_flow_style=False, sort_keys=False)
        return stream.getvalue()

    @patch('main.read_yaml_documents')  # Mock reading input and output files
    # Mock writing the modified input file
    @patch('builtins.open', new_callable=mock_open)
    @patch('main.yaml.safe_dump_all')  # Mock the dump for rewrite
    def test_verify_and_remove_success(self, mock_safe_dump_all, mock_open_write, mock_read_yaml):
        """Test successful verification and removal."""
        # Arrange
        output_path = Path('/output/rels.yaml')
        file1_path = Path('/in/file1.yaml')
        file2_path = Path('/in/file2.yaml')

        rel1 = {'kind': 'Relationship', 'name': 'r1', 'detail': 'abc'}
        rel2 = {'kind': 'Relationship', 'name': 'r2'}
        rel3 = {'kind': 'Relationship', 'name': 'r3'}
        table1 = {'kind': 'Table', 'name': 't1'}
        config1 = {'kind': 'Config', 'val': True}

        all_extracted_relationships = [
            rel1, rel2, rel3]  # As extracted initially
        original_relationships_by_file = {
            file1_path: [rel1, rel2],
            file2_path: [rel3]
        }

        # Simulate reading files:
        # 1. Read output file for verification
        # 2. Read input file 1 for removal
        # 3. Read input file 2 for removal
        output_file_content = [rel1, rel2, rel3]  # Content matches extracted
        input1_content = [table1, rel1, rel2]  # Original content
        input2_content = [rel3, config1]       # Original content

        read_call_count = 0
        read_map = {
            output_path: iter(output_file_content),
            file1_path: iter(input1_content),
            file2_path: iter(input2_content)
        }

        def read_side_effect(path):
            nonlocal read_call_count
            read_call_count += 1
            print(f"Mock Read: {path} ({read_call_count})")
            if path in read_map:
                # Important: re-create iterator for subsequent reads of the same file
                if path == output_path:
                    return iter(output_file_content)
                elif path == file1_path:
                    return iter(input1_content)
                elif path == file2_path:
                    return iter(input2_content)
            raise FileNotFoundError(
                f"Unexpected path in mock_read_yaml: {path}")
        mock_read_yaml.side_effect = read_side_effect

        # Act
        success = main.verify_and_remove(
            output_path, original_relationships_by_file, all_extracted_relationships)

        # Assert
        self.assertTrue(success)

        # Check reads: 1 for output, 1 for file1, 1 for file2
        self.assertEqual(mock_read_yaml.call_count, 3)
        mock_read_yaml.assert_has_calls([
            call(output_path),  # First verification read
            call(file1_path),  # Read for removal
            call(file2_path)   # Read for removal
        ], any_order=False)  # Ensure order

        # Check writes (rewrites of input files)
        # File 1 should be rewritten without rel1, rel2 -> keep [table1]
        # File 2 should be rewritten without rel3 -> keep [config1]
        self.assertEqual(mock_open_write.call_count, 2)
        mock_open_write.assert_has_calls([
            call(file1_path, 'w'),
            call(file2_path, 'w')
        ], any_order=True)

        # Check content dumped to files
        self.assertEqual(mock_safe_dump_all.call_count, 2)
        # Call args list is [(args1, kwargs1), (args2, kwargs2)]
        # Extract the first positional argument (the data list) from each call
        dumped_data_calls = [c.args[0]
                             for c in mock_safe_dump_all.call_args_list]

        # Check that the expected data was dumped, order doesn't matter
        self.assertIn([table1], dumped_data_calls)
        self.assertIn([config1], dumped_data_calls)

    @patch('main.read_yaml_documents')
    def test_verify_and_remove_verification_fail_count(self, mock_read_yaml):
        """Test verification failure due to count mismatch."""
        # Arrange
        output_path = Path('/output/rels.yaml')
        file1_path = Path('/in/file1.yaml')
        rel1 = {'kind': 'Relationship', 'name': 'r1'}
        all_extracted_relationships = [rel1]  # Expected 1
        original_relationships_by_file = {file1_path: [rel1]}

        # Simulate output file having 0 or 2 relationships
        mock_read_yaml.return_value = iter([])  # Output has 0

        # Act
        success = main.verify_and_remove(
            output_path, original_relationships_by_file, all_extracted_relationships)

        # Assert
        self.assertFalse(success)
        mock_read_yaml.assert_called_once_with(output_path)

    @patch('main.read_yaml_documents')
    def test_verify_and_remove_verification_fail_content(self, mock_read_yaml):
        """Test verification failure due to content mismatch."""
        # Arrange
        output_path = Path('/output/rels.yaml')
        file1_path = Path('/in/file1.yaml')
        rel1 = {'kind': 'Relationship', 'name': 'r1'}
        rel2_diff = {'kind': 'Relationship',
                     'name': 'r2_DIFFERENT'}  # Different content
        all_extracted_relationships = [rel1]
        original_relationships_by_file = {file1_path: [rel1]}

        # Simulate output file having different relationship
        mock_read_yaml.return_value = iter([rel2_diff])

        # Act
        success = main.verify_and_remove(
            output_path, original_relationships_by_file, all_extracted_relationships)

        # Assert
        self.assertFalse(success)
        mock_read_yaml.assert_called_once_with(output_path)

    @patch('main.read_yaml_documents')
    # Mock write failing
    @patch('builtins.open', side_effect=IOError("Write failed"))
    def test_verify_and_remove_rewrite_fail(self, mock_open_write, mock_read_yaml):
        """Test failure during the rewrite of an input file."""
        # Arrange (Similar setup to success case, but mock write fails)
        output_path = Path('/output/rels.yaml')
        file1_path = Path('/in/file1.yaml')
        rel1 = {'kind': 'Relationship', 'name': 'r1'}
        table1 = {'kind': 'Table', 'name': 't1'}
        all_extracted_relationships = [rel1]
        original_relationships_by_file = {file1_path: [rel1]}

        output_file_content = [rel1]
        input1_content = [table1, rel1]

        read_call_count = 0

        def read_side_effect(path):
            nonlocal read_call_count
            read_call_count += 1
            if path == output_path:
                return iter(output_file_content)
            elif path == file1_path:
                return iter(input1_content)
            else:
                raise FileNotFoundError()
        mock_read_yaml.side_effect = read_side_effect

        # Act
        success = main.verify_and_remove(
            output_path, original_relationships_by_file, all_extracted_relationships)

        # Assert
        self.assertFalse(success)
        mock_read_yaml.assert_has_calls([call(output_path), call(file1_path)])
        mock_open_write.assert_called_once_with(
            file1_path, 'w')  # Attempted to open for write
        # yaml.safe_dump_all should not be called because open failed
        # (Assuming the exception happens on open, not during write itself)

    # --- Test main function integration (High level) ---

    # Decorators are applied bottom-up. Arguments must match in reverse order.
    @patch('main.parse_arguments')          # mock_parse_args (last arg)
    @patch('main.find_files')               # mock_find_files
    @patch('main.create_backups')           # mock_create_backups
    @patch('main.extract_relationships')    # mock_extract
    @patch('main.write_output_file')        # mock_write_output
    @patch('main.Path')                     # mock_path_constructor
    # mock_exists - Be careful patching class AND method
    @patch('main.Path.exists')
    @patch('main.remove_backups')           # mock_remove_backups
    @patch('main.verify_and_remove')        # mock_verify_remove (first arg)
    @patch('builtins.input', return_value='yes')  # Mock user confirmation
    def test_main_success_flow(self, mock_input, mock_verify_remove, mock_remove_backups, mock_exists, mock_path_constructor, mock_write_output, mock_extract, mock_create_backups, mock_find_files, mock_parse_args):
        """Test the main function happy path with removal."""
        # Arrange
        # Mock args for removal case
        mock_args = argparse.Namespace(
            input_patterns=['*.yaml'], output='out.yaml', remove_from_source=True)
        mock_parse_args.return_value = mock_args

        # Configure the mock Path object returned by the constructor for output
        mock_output_path_instance = MagicMock(spec=Path)
        mock_output_path_instance.exists = MagicMock(
            return_value=False)  # Mock exists on instance
        mock_output_path_instance.resolve = MagicMock(
            return_value=mock_output_path_instance)  # Mock resolve on instance
        # When Path('out.yaml') is called, return our instance
        mock_path_constructor.side_effect = lambda p: mock_output_path_instance if p == 'out.yaml' else MagicMock()

        file1 = Path('/in/file1.yaml')
        mock_find_files.return_value = [file1]

        backup1 = Path('/in/.backup_file1.yaml')
        mock_create_backups.return_value = [backup1]

        rel1 = {'kind': 'Relationship', 'name': 'r1'}
        all_rels = [rel1]
        rels_by_file = {file1: [rel1]}
        mock_extract.return_value = (all_rels, rels_by_file)

        mock_verify_remove.return_value = True  # Verification succeeds

        # Act
        main.main()

        # Assert
        mock_parse_args.assert_called_once()
        mock_input.assert_called_once()  # Check confirmation was asked

        # Check Path('out.yaml') construction and method calls
        # Check it was called for output path
        mock_path_constructor.assert_any_call('out.yaml')
        mock_output_path_instance.resolve.assert_called_once_with()
        mock_output_path_instance.exists.assert_called_once()

        mock_find_files.assert_called_once_with(mock_args.input_patterns)
        mock_extract.assert_called_once_with([file1])
        mock_write_output.assert_called_once_with(
            mock_output_path_instance, all_rels)
        mock_create_backups.assert_called_once_with(
            [file1])  # Backups created before verify
        mock_verify_remove.assert_called_once_with(
            mock_output_path_instance, rels_by_file, all_rels)
        mock_remove_backups.assert_called_once_with(
            [backup1])  # Backups removed on success

    @patch('main.parse_arguments')
    @patch('main.find_files')
    @patch('main.create_backups')
    @patch('main.extract_relationships')
    @patch('main.write_output_file')
    @patch('main.verify_and_remove')
    @patch('main.remove_backups')
    @patch('sys.exit')
    @patch('main.Path')                     # Patch Path constructor
    @patch('builtins.input', return_value='yes')  # Mock user confirmation
    def test_main_verification_fail_flow(self, mock_input, mock_path_constructor, mock_exit, mock_remove_backups, mock_verify_remove, mock_write_output, mock_extract, mock_create_backups, mock_find_files, mock_parse_args):
        """Test the main function when verification fails."""
        # Arrange
        # Fix: Ensure remove_from_source is set for this flow
        mock_args = argparse.Namespace(
            input_patterns=['*.yaml'], output='out.yaml', remove_from_source=True)
        mock_parse_args.return_value = mock_args

        # Mock Path object for output path
        mock_output_path = MagicMock(spec=Path)
        mock_output_path.exists = MagicMock(return_value=False)
        mock_output_path.resolve = MagicMock(return_value=mock_output_path)
        mock_path_constructor.side_effect = lambda p: mock_output_path if p == 'out.yaml' else MagicMock()

        file1 = Path('/in/file1.yaml')
        mock_find_files.return_value = [file1]
        backup1 = Path('/in/.backup_file1.yaml')
        mock_create_backups.return_value = [backup1]
        rel1 = {'kind': 'Relationship', 'name': 'r1'}
        all_rels = [rel1]
        rels_by_file = {file1: [rel1]}
        mock_extract.return_value = (all_rels, rels_by_file)
        mock_verify_remove.return_value = False  # Verification FAILS

        # Act
        main.main()

        # Assert
        mock_input.assert_called_once()  # Check confirmation was asked
        mock_write_output.assert_called_once_with(mock_output_path, all_rels)
        mock_create_backups.assert_called_once_with([file1])
        mock_verify_remove.assert_called_once_with(
            mock_output_path, rels_by_file, all_rels)
        # IMPORTANT: Check that backups are NOT removed on failure
        mock_remove_backups.assert_not_called()
        # Check that sys.exit was called
        mock_exit.assert_called_once_with(1)

    # Fix: Correct argument order and add missing ones
    @patch('main.parse_arguments')          # mock_parse_args (last)
    @patch('main.find_files')               # mock_find_files
    # @patch('main.create_backups')          # Not called in this flow
    @patch('main.extract_relationships')    # mock_extract
    # @patch('main.remove_backups')          # Not called in this flow
    @patch('main.Path')                     # mock_path_constructor
    # @patch('main.Path.exists')             # Not called if output=None
    # @patch('main.Path.resolve')            # Not called if output=None
    # mock_write_stream (used in stdout mode)
    @patch('main.write_to_stream')
    # @patch('main.write_output_file')       # Not called in stdout mode
    # @patch('main.verify_and_remove')       # Not called in stdout mode
    @patch('sys.stdout', new_callable=io.StringIO)  # Mock stdout
    def test_main_stdout_no_relationships_found(self, mock_stdout, mock_write_stream, mock_path_constructor, mock_extract, mock_find_files, mock_parse_args):
        """Test the main function in stdout mode when no relationships are found."""
        # Arrange
        # Mock args for stdout mode
        mock_args = argparse.Namespace(
            input_patterns=['*.yaml'], output=None, remove_from_source=False)
        mock_parse_args.return_value = mock_args

        file1 = Path('/in/file1.yaml')
        mock_find_files.return_value = [file1]

        # Fix: Ensure extract returns a tuple (list, dict) even if empty
        mock_extract.return_value = ([], {file1: []})  # No relationships found

        # Act
        main.main()

        # Assert
        mock_find_files.assert_called_once_with(mock_args.input_patterns)
        mock_extract.assert_called_once_with([file1])
        # write_to_stream should NOT be called if no relationships found
        mock_write_stream.assert_not_called()
        # Check logged output for no relationships found message? (Optional)
        # self.assertIn("No 'kind: Relationship' stanzas found", mock_stdout.getvalue()) # Requires log capture

    # Fix: Add test for file output mode with no relationships found
    @patch('main.parse_arguments')          # mock_parse_args (last)
    @patch('main.find_files')               # mock_find_files
    # @patch('main.create_backups')          # Not called if no relationships found
    @patch('main.extract_relationships')    # mock_extract
    # @patch('main.remove_backups')          # Not called if no relationships found
    @patch('main.Path')                     # mock_path_constructor
    # @patch('main.Path.exists')             # Mocked via mock_path_constructor instance
    # @patch('main.Path.resolve')            # Mocked via mock_path_constructor instance
    @patch('main.write_output_file')        # mock_write_output
    # @patch('main.verify_and_remove')       # Not called if no relationships found
    def test_main_file_output_no_relationships_found(self, mock_write_output, mock_path_constructor, mock_extract, mock_find_files, mock_parse_args):
        """Test main function, file output mode, when no relationships are found."""
        # Arrange
        mock_args = argparse.Namespace(
            input_patterns=['*.yaml'], output='out.yaml', remove_from_source=False)
        mock_parse_args.return_value = mock_args

        # Mock Path object for output path
        mock_output_path = MagicMock(spec=Path)
        mock_output_path.exists = MagicMock(return_value=False)
        mock_output_path.resolve = MagicMock(return_value=mock_output_path)
        mock_path_constructor.side_effect = lambda p: mock_output_path if p == 'out.yaml' else MagicMock()

        file1 = Path('/in/file1.yaml')
        mock_find_files.return_value = [file1]

        # Simulate extract finding nothing
        mock_extract.return_value = ([], {file1: []})

        # Act
        main.main()

        # Assert
        mock_find_files.assert_called_once_with(mock_args.input_patterns)
        mock_extract.assert_called_once_with([file1])
        # Check that write_output_file *is* called with an empty list
        # to ensure the output file is created/overwritten as empty
        mock_write_output.assert_called_once_with(mock_output_path, [])
        # Other mocks like create_backups, verify_and_remove should not be called

    # Clean up original test_main_no_relationships_found - split into stdout/file tests above
    # This test was trying to do file output mode but had wrong mocks/asserts
    # def test_main_no_relationships_found(...): --> REMOVED


if __name__ == '__main__':
    unittest.main()
