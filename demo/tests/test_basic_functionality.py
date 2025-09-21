"""
Basic functionality tests for the genome validation experiment.

Tests core functionality of each module to ensure proper operation.
"""

import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.sequence import transform_sequence_to_coordinates, calculate_s_entropy
from src.genome import transform_genome_to_coordinates
from src.network import create_precision_network
from src.search import search_genome_space, compress_with_validation


class TestSequenceTransformation(unittest.TestCase):
    """Test sequence S-entropy transformation."""

    def test_sequence_transformation(self):
        """Test basic sequence transformation."""
        test_sequence = "ATCGATCGATCGATCG"
        coords = transform_sequence_to_coordinates(test_sequence)

        self.assertIsNotNone(coords)
        self.assertEqual(len(coords.coordinates), 64)  # Default dimensions
        self.assertGreater(coords.s_entropy, 0)
        self.assertEqual(len(coords.sequence_hash), 64)  # SHA256 hash length

    def test_s_entropy_calculation(self):
        """Test S-entropy calculation."""
        test_sequence = "ATCGATCGATCGATCG"
        s_entropy = calculate_s_entropy(test_sequence)

        self.assertIsInstance(s_entropy, float)
        self.assertGreaterEqual(s_entropy, 0)

    def test_coordinate_consistency(self):
        """Test that same sequence produces same coordinates."""
        test_sequence = "ATCGATCGATCGATCG"
        coords1 = transform_sequence_to_coordinates(test_sequence)
        coords2 = transform_sequence_to_coordinates(test_sequence)

        np.testing.assert_array_equal(coords1.coordinates, coords2.coordinates)
        self.assertEqual(coords1.s_entropy, coords2.s_entropy)


class TestGenomeTransformation(unittest.TestCase):
    """Test genome S-entropy transformation."""

    def test_genome_transformation(self):
        """Test basic genome transformation."""
        test_genome = {
            'chr1': 'ATCGATCG' * 100,
            'chr2': 'GCTAGCTA' * 80
        }

        coords = transform_genome_to_coordinates(test_genome, "test_genome")

        self.assertIsNotNone(coords)
        self.assertEqual(coords.genome_id, "test_genome")
        self.assertEqual(len(coords.chromosomes), 2)
        self.assertGreater(coords.total_s_entropy, 0)
        self.assertGreater(coords.genome_size, 0)

    def test_chromosome_processing(self):
        """Test individual chromosome processing."""
        test_genome = {
            'chr1': 'ATCGATCG' * 50
        }

        coords = transform_genome_to_coordinates(test_genome, "single_chr_genome")

        self.assertEqual(len(coords.chromosomes), 1)
        chromosome = coords.chromosomes[0]
        self.assertEqual(chromosome.chromosome_id, 'chr1')
        self.assertGreater(chromosome.s_entropy, 0)
        self.assertEqual(chromosome.length, 400)  # 8 * 50


class TestDistributedNetwork(unittest.TestCase):
    """Test distributed genomics network."""

    def test_network_creation(self):
        """Test network creation and basic operations."""
        network = create_precision_network()

        self.assertIsNotNone(network)

        # Test joining network
        node_id = network.join_network()
        self.assertIsNotNone(node_id)
        self.assertIn(node_id, network.active_nodes)

        # Test network status
        status = network.get_network_status()
        self.assertEqual(status['active_nodes'], 1)

        # Test leaving network
        success = network.leave_network(node_id)
        self.assertTrue(success)
        self.assertNotIn(node_id, network.active_nodes)

    def test_session_creation(self):
        """Test processing session creation."""
        network = create_precision_network()

        # Join nodes
        node_ids = []
        for i in range(3):
            node_id = network.join_network()
            node_ids.append(node_id)

        # Create session
        query_data = {'sequence': 'ATCGATCGATCGATCG'}
        session_id = network.create_processing_session(query_data)

        self.assertIsNotNone(session_id)
        self.assertIn(session_id, network.active_sessions)

        # Clean up
        for node_id in node_ids:
            network.leave_network(node_id)


class TestSearchAlgorithms(unittest.TestCase):
    """Test Graffiti-based search algorithms."""

    def test_sequence_search(self):
        """Test sequence-based search."""
        results = search_genome_space(
            query="ATCGATCGATCGATCG",
            query_type="sequence",
            max_results=3
        )

        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)

        if results:
            result = results[0]
            self.assertGreater(result.relevance_score, 0)
            self.assertIsNotNone(result.genomic_coordinates)
            self.assertIsInstance(result.llm_reasoning, str)

    def test_compression_validation(self):
        """Test proof-validated compression."""
        test_data = {
            'sequence': 'ATCGATCGATCGATCG',
            'metadata': {'test': True}
        }

        compressed_data, proof_validation = compress_with_validation(test_data)

        self.assertIsNotNone(compressed_data)
        self.assertIsInstance(proof_validation, dict)
        self.assertIn('compression_ratio', proof_validation)
        self.assertIn('proof_verified', proof_validation)


class TestSystemIntegration(unittest.TestCase):
    """Test system integration."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Step 1: Transform sequence
        test_sequence = "ATCGATCGATCGATCG"
        seq_coords = transform_sequence_to_coordinates(test_sequence)

        # Step 2: Create network and search
        network = create_precision_network()
        node_id = network.join_network()

        # Step 3: Search genome space
        results = search_genome_space(
            query=test_sequence,
            query_type="sequence",
            network=network,
            max_results=1
        )

        # Verify workflow
        self.assertIsNotNone(seq_coords)
        self.assertIsInstance(results, list)

        # Clean up
        network.leave_network(node_id)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
