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
logging.disable(logging.CRITICAL)


class TestYamlRelationshipExtractor(unittest.TestCase):

    @patch('main.argparse.ArgumentParser')
    def test_parse_arguments(self, mock_parser_class):
        """Test argument parsing."""
        # Arrange
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = argparse.Namespace(
            input_patterns=['input/*.yaml', 'another/dir/**/*.yml'],
            output='output/relationships.yaml'
        )
        mock_parser_class.return_value = mock_parser_instance

        # Act
        args = main.parse_arguments()

        # Assert
        self.assertEqual(args.input_patterns, [
                         'input/*.yaml', 'another/dir/**/*.yml'])
        self.assertEqual(args.output, 'output/relationships.yaml')
        mock_parser_instance.add_argument.assert_any_call(
            'input_patterns', nargs='+', help=unittest.mock.ANY)
        mock_parser_instance.add_argument.assert_any_call(
            '-o', '--output', required=True, help=unittest.mock.ANY)
        mock_parser_instance.parse_args.assert_called_once()

    @patch('main.glob.glob')
    @patch('main.Path.is_file')
    @patch('main.Path.resolve')  # Mock resolve to control path generation
    # Simple pass-through for expanduser
    @patch('os.path.expanduser', side_effect=lambda x: x)
    def test_find_files(self, mock_expanduser, mock_resolve, mock_is_file, mock_glob):
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

        # Simulate Path object behavior
        # Need to create mock Path objects returned by resolve()
        path_a = MagicMock(spec=Path)
        path_a.name = 'a.yaml'
        path_a.__str__.return_value = '/abs/path/a.yaml'

        path_b = MagicMock(spec=Path)
        path_b.name = 'b.yaml'
        path_b.__str__.return_value = '/abs/path/b.yaml'

        path_c1 = MagicMock(spec=Path)
        path_c1.name = 'c.yaml'
        path_c1.__str__.return_value = '/abs/path/data/sub/c.yaml'

        path_c2 = MagicMock(spec=Path)
        path_c2.name = 'c.yaml'
        path_c2.__str__.return_value = '/abs/path/data/c.yaml'

        path_dir = MagicMock(spec=Path)
        path_dir.name = 'a_dir'
        path_dir.__str__.return_value = '/abs/path/a_dir'

        # Link resolve results to Path objects
        resolve_map = {
            'a.yaml': path_a,
            'b.yaml': path_b,
            'data/sub/c.yaml': path_c1,
            'data/c.yaml': path_c2,
            'a_dir': path_dir
        }
        # p is the Path object itself
        mock_resolve.side_effect = lambda p: resolve_map[str(p)]

        # Define which paths are files
        isfile_map = {
            path_a: True,
            path_b: True,
            path_c1: True,
            path_c2: True,
            path_dir: False  # a_dir is not a file
        }
        mock_is_file.side_effect = lambda p: isfile_map.get(p, False)

        # Map Path calls in find_files back to strings for resolve_map
        # This requires patching Path constructor or how it's used internally
        # Easier: Mock Path() directly for the file strings returned by glob
        path_init_map = {
            'a.yaml': path_a,
            'b.yaml': path_b,
            'data/sub/c.yaml': path_c1,
            'data/c.yaml': path_c2,
            'a_dir': path_dir
        }

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
            self.assertEqual(mock_path_constructor.call_count,
                             5)  # a, b, c1, c2, a_dir
            # Resolve is called within Path(f).resolve()
            # mock_resolve should have been called 5 times, let's check Path.resolve()
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

            # Check final list (sorted, absolute paths represented by mocks)
            # Sort expected list based on string representation for comparison
            expected_files = sorted(
                [path_a, path_b, path_c1, path_c2], key=str)
            # Sort actual list the same way for comparison
            self.assertEqual(sorted(found_files, key=str), expected_files)

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

        def read_side_effect(path):
            nonlocal read_call_count
            read_call_count += 1
            if path == output_path:
                print(f"Mock Read: Output ({read_call_count})")
                return iter(output_file_content)
            elif path == file1_path:
                print(f"Mock Read: Input 1 ({read_call_count})")
                return iter(input1_content)
            elif path == file2_path:
                print(f"Mock Read: Input 2 ({read_call_count})")
                return iter(input2_content)
            else:
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
        # Get the file handles passed to dump_all
        # The file handles mock_open returns
        handles = [c.args[0] for c in mock_open_write.call_args_list]

        # Check calls to yaml.safe_dump_all (might be tricky with mock handles)
        # We check what *should* have been dumped
        self.assertEqual(mock_safe_dump_all.call_count, 2)
        # Check call arguments (expected data to write)
        # Order depends on dict iteration order, which is stable for these tests
        call_args_list = mock_safe_dump_all.call_args_list

        # Find which call corresponds to which file based on what was kept
        dump_args_file1 = None
        dump_args_file2 = None
        for dump_call in call_args_list:
            data_dumped = dump_call.args[0]
            if data_dumped == [table1]:
                dump_args_file1 = dump_call
            elif data_dumped == [config1]:
                dump_args_file2 = dump_call

        self.assertIsNotNone(
            dump_args_file1, "Write call for file1 not found or data incorrect")
        self.assertIsNotNone(
            dump_args_file2, "Write call for file2 not found or data incorrect")

        # Verify other dump parameters
        self.assertEqual(dump_args_file1.args[0], [table1])
        self.assertIsInstance(
            dump_args_file1.args[1], MagicMock)  # File handle
        self.assertEqual(dump_args_file1.kwargs, {
                         'default_flow_style': False, 'sort_keys': False})

        self.assertEqual(dump_args_file2.args[0], [config1])
        self.assertIsInstance(
            dump_args_file2.args[1], MagicMock)  # File handle
        self.assertEqual(dump_args_file2.kwargs, {
                         'default_flow_style': False, 'sort_keys': False})

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

    @patch('main.verify_and_remove')
    @patch('main.remove_backups')
    @patch('main.Path.exists')
    # Patch Path constructor instead of resolve directly
    @patch('main.Path')
    def test_main_success_flow(self, mock_path_constructor, mock_exists, mock_remove_backups, mock_verify_remove, mock_write_output, mock_extract, mock_create_backups, mock_find_files, mock_parse_args):
        """Test the main function happy path."""
        # Arrange
        mock_args = argparse.Namespace(
            input_patterns=['*.yaml'], output='out.yaml')
        mock_parse_args.return_value = mock_args

        # Configure the mock Path object returned by the constructor
        mock_output_path_instance = MagicMock(spec=Path)
        mock_path_constructor.return_value = mock_output_path_instance

        # Mock the exists method on the instance returned by Path('out.yaml')
        mock_output_path_instance.exists.return_value = False

        # Mock the resolve method on the instance returned by Path('out.yaml')
        # It should return itself after resolving (or a new mock if paths change)
        mock_output_path_instance.resolve.return_value = mock_output_path_instance

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

        # Check Path('out.yaml') was called
        mock_path_constructor.assert_called_once_with('out.yaml')
        # Check resolve() was called on the instance
        mock_output_path_instance.resolve.assert_called_once_with()
        # Check exists() was called on the instance
        mock_output_path_instance.exists.assert_called_once()

        mock_find_files.assert_called_once_with(mock_args.input_patterns)
        mock_create_backups.assert_called_once_with([file1])
        mock_extract.assert_called_once_with([file1])
        mock_write_output.assert_called_once_with(
            mock_output_path_instance, all_rels)  # Use the instance
        mock_verify_remove.assert_called_once_with(
            mock_output_path_instance, rels_by_file, all_rels)  # Use the instance
        # Check that backups are removed on success
        mock_remove_backups.assert_called_once_with([backup1])

    @patch('main.parse_arguments')
    @patch('main.find_files')
    @patch('main.create_backups')
    @patch('main.extract_relationships')
    @patch('main.write_output_file')
    @patch('main.verify_and_remove')
    @patch('main.remove_backups')
    @patch('sys.exit')
    @patch('main.Path.exists')
    @patch('main.Path.resolve')
    def test_main_verification_fail_flow(self, mock_resolve, mock_exists, mock_exit, mock_remove_backups, mock_verify_remove, mock_write_output, mock_extract, mock_create_backups, mock_find_files, mock_parse_args):
        """Test the main function when verification fails."""
        # Arrange
        mock_args = argparse.Namespace(
            input_patterns=['*.yaml'], output='out.yaml')
        mock_parse_args.return_value = mock_args
        mock_output_path = MagicMock(spec=Path)
        mock_resolve.return_value = mock_output_path
        mock_exists.return_value = False
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
        # Check execution up to verify_and_remove
        mock_write_output.assert_called_once_with(mock_output_path, all_rels)
        mock_verify_remove.assert_called_once_with(
            mock_output_path, rels_by_file, all_rels)
        # IMPORTANT: Check that backups are NOT removed on failure
        mock_remove_backups.assert_not_called()
        # Check that sys.exit was called
        mock_exit.assert_called_once_with(1)

    @patch('main.parse_arguments')
    @patch('main.find_files')
    @patch('main.create_backups')
    @patch('main.extract_relationships')
    @patch('main.remove_backups')  # Need to mock remove_backups here too
    @patch('main.Path.exists')
    @patch('main.Path.resolve')
    @patch('main.write_output_file')  # Added patch
    @patch('main.verify_and_remove')  # Added patch
    def test_main_no_relationships_found(self, mock_resolve, mock_exists, mock_remove_backups, mock_extract, mock_create_backups, mock_find_files, mock_parse_args, mock_write_output, mock_verify_remove):  # Added mock args
        """Test the main function when no relationships are found."""
        # Arrange
        mock_args = argparse.Namespace(
            input_patterns=['*.yaml'], output='out.yaml')
        mock_parse_args.return_value = mock_args
        mock_output_path = MagicMock(spec=Path)
        mock_resolve.return_value = mock_output_path
        mock_exists.return_value = False
        file1 = Path('/in/file1.yaml')
        mock_find_files.return_value = [file1]
        backup1 = Path('/in/.backup_file1.yaml')
        mock_create_backups.return_value = [backup1]

        # Simulate extract finding nothing
        mock_extract.return_value = ([], {file1: []})

        # Act
        main.main()

        # Assert
        mock_extract.assert_called_once_with([file1])
        # Check that backups ARE removed when nothing is found (clean exit)
        mock_remove_backups.assert_called_once_with([backup1])
        # write_output and verify_and_remove should not be called
        mock_write_output.assert_not_called()
        mock_verify_remove.assert_not_called()


if __name__ == '__main__':
    unittest.main()
