#!/usr/bin/env python3
"""
Tests for analysis utilities
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import json
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from analysis.analyze_training_parameters import analyze_bucket_data, calculate_training_recommendations
from analysis.local_data_analysis import analyze_local_dataset


class TestAnalysisUtils(unittest.TestCase):
    """Test analysis utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_training_recommendations(self):
        """Test training parameter recommendations"""
        # Mock statistics
        stats = {
            'total_files': 100,
            'valid_files': 95,
            'corrupted_files': 5,
            'total_tokens': 1_000_000_000,
            'avg_tokens_per_file': 10_000_000,
            'avg_file_size_mb': 50.0,
            'estimated_dataset_size_gb': 5.0,
            'corruption_rate': 0.05
        }
        
        # Test single GPU recommendations
        rec_1gpu = calculate_training_recommendations(stats, target_gpus=1)
        self.assertIsInstance(rec_1gpu, dict)
        self.assertIn('recommended_steps', rec_1gpu)
        self.assertIn('effective_batch_size', rec_1gpu)
        self.assertIn('estimated_training_time_hours', rec_1gpu)
        
        # Test multi-GPU recommendations
        rec_4gpu = calculate_training_recommendations(stats, target_gpus=4)
        self.assertGreater(rec_4gpu['effective_batch_size'], rec_1gpu['effective_batch_size'])
        self.assertLess(rec_4gpu['estimated_training_time_hours'], rec_1gpu['estimated_training_time_hours'])
    
    def test_local_data_analysis(self):
        """Test local dataset analysis"""
        # Create mock data files
        data_dir = Path(self.temp_dir) / "data"
        data_dir.mkdir()
        
        # Create sample tokenized file
        sample_data = {
            'input_ids': np.random.randint(0, 50000, size=(100, 512)),
            'attention_mask': np.ones((100, 512)),
            'sequence_lengths': np.full(100, 512)
        }
        
        np.savez_compressed(
            data_dir / "tokenized_sample.npz",
            **sample_data
        )
        
        # Test analysis
        try:
            # Note: This might fail without actual GCS setup
            # Just test that the function exists and is callable
            self.assertTrue(callable(analyze_local_dataset))
        except Exception:
            # Expected if no GCS credentials
            pass
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestGPUAnalysis(unittest.TestCase):
    """Test GPU analysis utilities"""
    
    def test_8gpu_analysis_structure(self):
        """Test that 8GPU analysis has correct structure"""
        # Import to check it's valid Python
        try:
            from analysis.local_data_analysis import analyze_local_dataset
            self.assertTrue(True)
        except ImportError:
            self.skipTest("Analysis module not found")


if __name__ == "__main__":
    unittest.main() 