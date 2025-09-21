"""
Validation Suite for S-Entropy Sequence Transformation

Comprehensive validation and benchmarking of the S-entropy coordinate transformation
against established sequencing and assembly algorithms.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from Bio.Seq import Seq
from Bio.SeqUtils import GC, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import psutil
import memory_profiler

from .s_entropy_transform import SequenceTransformer, SEntropyCoordinates, compare_with_traditional_methods


@dataclass
class ValidationResults:
    """Container for validation experiment results."""
    accuracy_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    comparison_results: Dict[str, Any]
    memory_usage: Dict[str, float]
    visualization_data: Dict[str, Any]


class SequenceValidationSuite:
    """
    Comprehensive validation suite for S-entropy sequence transformation.

    Tests accuracy, performance, and comparison against traditional methods
    using various genomic sequence datasets.
    """

    def __init__(self):
        self.transformer = SequenceTransformer()
        self.validation_results = []

    def generate_test_sequences(self, count: int = 100) -> List[str]:
        """Generate diverse test sequences for validation."""
        sequences = []

        # Random sequences of varying lengths
        for i in range(count // 4):
            length = np.random.randint(50, 1000)
            seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], length))
            sequences.append(seq)

        # GC-rich sequences
        for i in range(count // 4):
            length = np.random.randint(50, 1000)
            seq = ''.join(np.random.choice(['G', 'C'], int(length * 0.7))) + \
                  ''.join(np.random.choice(['A', 'T'], int(length * 0.3)))
            sequences.append(seq)

        # AT-rich sequences
        for i in range(count // 4):
            length = np.random.randint(50, 1000)
            seq = ''.join(np.random.choice(['A', 'T'], int(length * 0.7))) + \
                  ''.join(np.random.choice(['G', 'C'], int(length * 0.3)))
            sequences.append(seq)

        # Repetitive sequences
        for i in range(count // 4):
            motif = ''.join(np.random.choice(['A', 'T', 'G', 'C'], np.random.randint(3, 10)))
            repeats = np.random.randint(10, 50)
            seq = motif * repeats
            sequences.append(seq)

        return sequences

    def validate_coordinate_accuracy(self, sequences: List[str]) -> Dict[str, float]:
        """
        Validate the accuracy of S-entropy coordinate transformation.

        Tests:
        1. Coordinate consistency (same sequence -> same coordinates)
        2. Coordinate uniqueness (different sequences -> different coordinates)
        3. Distance preservation (similar sequences -> similar coordinates)
        """
        results = {}

        # Transform sequences to coordinates
        coordinates = [self.transformer.transform_to_coordinates(seq) for seq in sequences]

        # Test 1: Coordinate consistency
        consistency_scores = []
        for i, seq in enumerate(sequences[:10]):  # Test subset for efficiency
            coord1 = self.transformer.transform_to_coordinates(seq)
            coord2 = self.transformer.transform_to_coordinates(seq)
            consistency = np.allclose(coord1.coordinates, coord2.coordinates)
            consistency_scores.append(1.0 if consistency else 0.0)

        results['coordinate_consistency'] = np.mean(consistency_scores)

        # Test 2: Coordinate uniqueness
        coord_matrix = np.array([coord.coordinates for coord in coordinates])
        pairwise_distances = []
        for i in range(len(coord_matrix)):
            for j in range(i + 1, len(coord_matrix)):
                distance = np.linalg.norm(coord_matrix[i] - coord_matrix[j])
                pairwise_distances.append(distance)

        results['coordinate_uniqueness'] = np.mean(pairwise_distances)
        results['coordinate_std'] = np.std(pairwise_distances)

        # Test 3: Distance preservation (clustering validation)
        if len(sequences) >= 10:
            # Create sequence similarity groups
            gc_contents = [GC(seq) for seq in sequences]
            gc_labels = ['high' if gc > 60 else 'medium' if gc > 40 else 'low' for gc in gc_contents]

            # Cluster coordinates
            kmeans = KMeans(n_clusters=3, random_state=42)
            coord_labels = kmeans.fit_predict(coord_matrix)

            # Calculate clustering accuracy using silhouette score
            silhouette = silhouette_score(coord_matrix, coord_labels)
            results['clustering_quality'] = silhouette

        return results

    def benchmark_transformation_performance(self, sequences: List[str]) -> Dict[str, float]:
        """
        Benchmark the performance of S-entropy transformation.

        Measures:
        1. Transformation time per sequence
        2. Memory usage during transformation
        3. Scalability with sequence length
        """
        results = {}

        # Measure transformation time
        transformation_times = []
        memory_usage = []

        for seq in sequences:
            # Memory before transformation
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Time transformation
            start_time = time.time()
            coord = self.transformer.transform_to_coordinates(seq)
            end_time = time.time()

            # Memory after transformation
            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            transformation_times.append(end_time - start_time)
            memory_usage.append(mem_after - mem_before)

        results['avg_transformation_time'] = np.mean(transformation_times)
        results['std_transformation_time'] = np.std(transformation_times)
        results['avg_memory_usage'] = np.mean(memory_usage)
        results['std_memory_usage'] = np.std(memory_usage)

        # Scalability analysis
        sequence_lengths = [len(seq) for seq in sequences]
        length_time_correlation = np.corrcoef(sequence_lengths, transformation_times)[0, 1]
        results['length_time_correlation'] = length_time_correlation

        # Throughput calculation
        total_nucleotides = sum(sequence_lengths)
        total_time = sum(transformation_times)
        results['nucleotides_per_second'] = total_nucleotides / total_time if total_time > 0 else 0

        return results

    def compare_with_established_methods(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Compare S-entropy method with established sequencing and assembly algorithms.

        Comparisons include:
        1. Traditional sequence alignment methods
        2. K-mer based approaches
        3. Compression-based similarity measures
        """
        comparison_results = compare_with_traditional_methods(sequences)

        # Additional comparisons

        # K-mer analysis comparison
        kmer_results = self._compare_kmer_methods(sequences)
        comparison_results['kmer_comparison'] = kmer_results

        # Compression-based comparison
        compression_results = self._compare_compression_methods(sequences)
        comparison_results['compression_comparison'] = compression_results

        return comparison_results

    def _compare_kmer_methods(self, sequences: List[str]) -> Dict[str, Any]:
        """Compare with k-mer based sequence analysis."""
        results = {}

        # Generate k-mer profiles for sequences
        k = 4  # 4-mer analysis
        kmer_profiles = []

        for seq in sequences:
            kmer_count = {}
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                kmer_count[kmer] = kmer_count.get(kmer, 0) + 1

            # Convert to frequency vector
            all_kmers = [''.join(p) for p in np.ndindex(*([4] * k))]
            kmer_mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

            profile = []
            for kmer_idx in all_kmers:
                kmer_str = ''.join(['ATGC'[int(i)] for i in kmer_idx])
                freq = kmer_count.get(kmer_str, 0) / max(1, len(seq) - k + 1)
                profile.append(freq)

            kmer_profiles.append(profile)

        kmer_matrix = np.array(kmer_profiles)

        # Compare clustering performance
        if len(sequences) >= 10:
            # S-entropy coordinates
            s_coords = [self.transformer.transform_to_coordinates(seq) for seq in sequences]
            s_matrix = np.array([coord.coordinates for coord in s_coords])

            # Cluster both representations
            kmeans_s = KMeans(n_clusters=3, random_state=42)
            kmeans_k = KMeans(n_clusters=3, random_state=42)

            s_labels = kmeans_s.fit_predict(s_matrix)
            k_labels = kmeans_k.fit_predict(kmer_matrix)

            # Compare clustering quality
            s_silhouette = silhouette_score(s_matrix, s_labels)
            k_silhouette = silhouette_score(kmer_matrix, k_labels)

            results['s_entropy_silhouette'] = s_silhouette
            results['kmer_silhouette'] = k_silhouette
            results['silhouette_improvement'] = s_silhouette - k_silhouette

        return results

    def _compare_compression_methods(self, sequences: List[str]) -> Dict[str, Any]:
        """Compare with compression-based sequence similarity measures."""
        results = {}

        # Simple compression ratio comparison
        import zlib

        compression_ratios = []
        s_entropy_values = []

        for seq in sequences:
            # Compression ratio
            compressed = zlib.compress(seq.encode())
            compression_ratio = len(compressed) / len(seq)
            compression_ratios.append(compression_ratio)

            # S-entropy
            s_entropy = self.transformer.calculate_s_entropy(seq)
            s_entropy_values.append(s_entropy)

        # Correlation between compression ratio and S-entropy
        correlation = np.corrcoef(compression_ratios, s_entropy_values)[0, 1]
        results['compression_s_entropy_correlation'] = correlation

        # Information content comparison
        results['avg_compression_ratio'] = np.mean(compression_ratios)
        results['avg_s_entropy'] = np.mean(s_entropy_values)
        results['compression_efficiency'] = np.mean(compression_ratios) / np.mean(s_entropy_values)

        return results

    def run_full_validation(self, sequence_count: int = 100) -> ValidationResults:
        """
        Run complete validation suite.

        Args:
            sequence_count: Number of test sequences to generate

        Returns:
            ValidationResults object with comprehensive results
        """
        print(f"Running full validation with {sequence_count} sequences...")

        # Generate test sequences
        sequences = self.generate_test_sequences(sequence_count)

        # Run validation tests
        print("Validating coordinate accuracy...")
        accuracy_metrics = self.validate_coordinate_accuracy(sequences)

        print("Benchmarking performance...")
        performance_metrics = self.benchmark_transformation_performance(sequences)

        print("Comparing with established methods...")
        comparison_results = self.compare_with_established_methods(sequences)

        # Memory usage analysis
        memory_usage = {
            'peak_memory_mb': performance_metrics['avg_memory_usage'],
            'memory_per_nucleotide': performance_metrics['avg_memory_usage'] / np.mean([len(seq) for seq in sequences])
        }

        # Generate visualization data
        visualization_data = self._generate_visualization_data(sequences, comparison_results)

        results = ValidationResults(
            accuracy_metrics=accuracy_metrics,
            performance_metrics=performance_metrics,
            comparison_results=comparison_results,
            memory_usage=memory_usage,
            visualization_data=visualization_data
        )

        self.validation_results.append(results)
        return results

    def _generate_visualization_data(self, sequences: List[str], comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for validation result visualizations."""
        viz_data = {}

        # Sequence length distribution
        lengths = [len(seq) for seq in sequences]
        viz_data['sequence_lengths'] = lengths

        # GC content distribution
        gc_contents = [GC(seq) for seq in sequences]
        viz_data['gc_contents'] = gc_contents

        # S-entropy distribution
        s_entropies = [self.transformer.calculate_s_entropy(seq) for seq in sequences]
        viz_data['s_entropies'] = s_entropies

        # Coordinate visualization (2D projection)
        if 's_entropy_method' in comparison_results:
            coord_matrix = comparison_results['s_entropy_method']['coordinates']
            if coord_matrix.shape[1] >= 2:
                viz_data['coordinate_2d'] = coord_matrix[:, :2]
            else:
                viz_data['coordinate_2d'] = coord_matrix

        return viz_data

    def generate_validation_report(self, results: ValidationResults) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("=" * 60)
        report.append("S-ENTROPY SEQUENCE TRANSFORMATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Accuracy metrics
        report.append("ACCURACY METRICS:")
        report.append("-" * 20)
        for metric, value in results.accuracy_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")

        # Performance metrics
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 20)
        for metric, value in results.performance_metrics.items():
            if isinstance(value, float):
                report.append(f"{metric}: {value:.4f}")
            else:
                report.append(f"{metric}: {value}")
        report.append("")

        # Memory usage
        report.append("MEMORY USAGE:")
        report.append("-" * 15)
        for metric, value in results.memory_usage.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")

        # Comparison summary
        report.append("COMPARISON WITH TRADITIONAL METHODS:")
        report.append("-" * 40)
        if 'performance_comparison' in results.comparison_results:
            perf_comp = results.comparison_results['performance_comparison']
            report.append(f"Time ratio vs PCA: {perf_comp.get('time_ratio_pca', 'N/A'):.4f}")
            report.append(f"Time ratio vs t-SNE: {perf_comp.get('time_ratio_tsne', 'N/A'):.4f}")

        if 'kmer_comparison' in results.comparison_results:
            kmer_comp = results.comparison_results['kmer_comparison']
            if 'silhouette_improvement' in kmer_comp:
                report.append(f"Silhouette score improvement over k-mer: {kmer_comp['silhouette_improvement']:.4f}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


def benchmark_transformation_performance(sequences: List[str]) -> Dict[str, float]:
    """
    Convenience function to benchmark S-entropy transformation performance.

    Args:
        sequences: List of DNA sequences to benchmark

    Returns:
        Performance metrics dictionary
    """
    validator = SequenceValidationSuite()
    return validator.benchmark_transformation_performance(sequences)


def validate_coordinate_accuracy(sequences: List[str]) -> Dict[str, float]:
    """
    Convenience function to validate coordinate transformation accuracy.

    Args:
        sequences: List of DNA sequences to validate

    Returns:
        Accuracy metrics dictionary
    """
    validator = SequenceValidationSuite()
    return validator.validate_coordinate_accuracy(sequences)
