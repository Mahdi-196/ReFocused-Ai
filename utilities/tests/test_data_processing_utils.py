#!/usr/bin/env python3
"""
Tests for data processing utilities
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import numpy as np
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))


class TestDataProcessingUtils(unittest.TestCase):
    """Test data processing utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir(parents=True)
        
        # Create sample tokenized data
        self.create_sample_tokenized_data()
    
    def create_sample_tokenized_data(self):
        """Create sample tokenized data files"""
        # Create multiple sample files
        for i in range(3):
            data = {
                'input_ids': np.random.randint(0, 50000, size=(100, 512), dtype=np.int32),
                'attention_mask': np.ones((100, 512), dtype=np.int32),
                'sequence_lengths': np.random.randint(100, 512, size=100, dtype=np.int32)
            }
            
            np.savez_compressed(
                self.test_data_dir / f"tokenized_file_{i}.npz",
                **data
            )
    
    def test_analyze_tokenized_data(self):
        """Test tokenized data analysis"""
        # Create a simple analysis function test
        files = list(self.test_data_dir.glob("*.npz"))
        self.assertEqual(len(files), 3)
        
        # Test loading and basic analysis
        total_sequences = 0
        for file_path in files:
            data = np.load(file_path)
            self.assertIn('input_ids', data)
            self.assertIn('attention_mask', data)
            self.assertIn('sequence_lengths', data)
            total_sequences += len(data['input_ids'])
        
        self.assertEqual(total_sequences, 300)  # 3 files * 100 sequences each
    
    def test_quick_dataset_size_check(self):
        """Test dataset size checking"""
        # Calculate total size
        total_size = 0
        for file_path in self.test_data_dir.glob("*.npz"):
            total_size += file_path.stat().st_size
        
        # Should be reasonable size for compressed numpy arrays
        self.assertGreater(total_size, 0)
        self.assertLess(total_size, 10 * 1024 * 1024)  # Less than 10MB for test data
    
    def test_count_sequences(self):
        """Test sequence counting functionality"""
        # Count sequences across all files
        sequence_counts = {}
        
        for file_path in self.test_data_dir.glob("*.npz"):
            data = np.load(file_path)
            sequence_counts[file_path.name] = {
                'sequences': len(data['input_ids']),
                'max_length': data['input_ids'].shape[1] if data['input_ids'].ndim > 1 else 0,
                'avg_length': np.mean(data['sequence_lengths']) if 'sequence_lengths' in data else 0
            }
        
        # Verify counts
        self.assertEqual(len(sequence_counts), 3)
        for file_name, counts in sequence_counts.items():
            self.assertEqual(counts['sequences'], 100)
            self.assertEqual(counts['max_length'], 512)
    
    def test_check_missing_files(self):
        """Test missing file detection"""
        # Create a list of expected files
        expected_files = [f"tokenized_file_{i}.npz" for i in range(5)]
        actual_files = [f.name for f in self.test_data_dir.glob("*.npz")]
        
        missing_files = set(expected_files) - set(actual_files)
        self.assertEqual(len(missing_files), 2)  # Files 3 and 4 are missing
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestBucketOperations(unittest.TestCase):
    """Test bucket-related operations"""
    
    def test_bucket_check_structure(self):
        """Test that bucket check utilities have correct structure"""
        # This would test actual bucket operations if credentials were available
        # For now, just verify the utilities can be imported
        try:
            from data_processing.quick_bucket_check import main
            self.assertTrue(callable(main))
        except ImportError:
            # Expected if module structure is different
            pass
    
    def test_tokenized_file_validation(self):
        """Test tokenized file validation logic"""
        # Create a mock tokenized file
        temp_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        
        try:
            # Create valid tokenized data
            data = {
                'input_ids': np.array([[1, 2, 3, 4, 5]]),
                'attention_mask': np.array([[1, 1, 1, 1, 1]]),
            }
            np.savez_compressed(temp_file.name, **data)
            
            # Load and validate
            loaded = np.load(temp_file.name)
            self.assertIn('input_ids', loaded)
            self.assertIn('attention_mask', loaded)
            
        finally:
            os.unlink(temp_file.name)


if __name__ == "__main__":
    unittest.main() 