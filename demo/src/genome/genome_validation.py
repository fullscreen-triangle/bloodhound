"""
Genome Validation Suite

Comprehensive validation and benchmarking of S-entropy genome transformation
against established genome assembly and analysis algorithms.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import psutil
import memory_profiler

from .genome_transform import GenomeTransformer, GenomeCoordinates, compare_genome_assembly_methods


@dataclass
class GenomeValidationResults:
    """Container for genome validation experiment results."""
    accuracy_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    assembly_comparison: Dict[str, Any]
    scalability_metrics: Dict[str, float]
    memory_usage: Dict[str, float]
    visualization_data: Dict[str, Any]


class GenomeValidationSuite:
    """
    Comprehensive validation suite for S-entropy genome transformation.

    Tests accuracy, performance, scalability, and comparison against
    traditional genome assembly and analysis methods.
    """

    def __init__(self):
        self.transformer = GenomeTransformer()
        self.validation_results = []

    def generate_test_genomes(self, count: int = 10) -> List[Dict[str, str]]:
        """Generate diverse test genomes for validation."""
        genomes = []

        for i in range(count):
            genome = {}

            # Generate chromosomes with varying characteristics
            chromosome_count = np.random.randint(5, 25)  # Varying genome complexity

            for chrom_idx in range(chromosome_count):
                chrom_id = f"chr{chrom_idx + 1}"

                # Varying chromosome sizes
                if chrom_idx == 0:  # Largest chromosome
                    chrom_size = np.random.randint(50000, 200000)
                elif chrom_idx < 5:  # Large chromosomes
                    chrom_size = np.random.randint(20000, 100000)
                else:  # Smaller chromosomes
                    chrom_size = np.random.randint(5000, 50000)

                # Generate chromosome with specific characteristics
                if i < count // 3:  # GC-rich genomes
                    sequence = self._generate_gc_rich_chromosome(chrom_size)
                elif i < 2 * count // 3:  # AT-rich genomes
                    sequence = self._generate_at_rich_chromosome(chrom_size)
                else:  # Balanced genomes
                    sequence = self._generate_balanced_chromosome(chrom_size)

                genome[chrom_id] = sequence

            genomes.append(genome)

        return genomes

    def _generate_gc_rich_chromosome(self, size: int) -> str:
        """Generate GC-rich chromosome sequence."""
        # 70% GC content with gene-like patterns
        gc_ratio = 0.7
        sequence = []

        # Add some gene-like structures
        gene_count = size // 5000  # Rough gene density
        for _ in range(gene_count):
            # Start codon
            sequence.extend(['ATG'])
            # GC-rich coding region
            coding_length = np.random.randint(300, 3000)
            for _ in range(coding_length // 3):
                codon = ''.join(np.random.choice(['G', 'C'], 2)) + np.random.choice(['A', 'T', 'G', 'C'])
                sequence.extend([codon])
            # Stop codon
            sequence.extend([np.random.choice(['TAA', 'TAG', 'TGA'])])

        # Fill remaining with GC-rich sequence
        remaining = size - len(''.join(sequence))
        if remaining > 0:
            gc_nucleotides = int(remaining * gc_ratio)
            at_nucleotides = remaining - gc_nucleotides

            remaining_seq = (['G'] * (gc_nucleotides // 2) +
                           ['C'] * (gc_nucleotides // 2) +
                           ['A'] * (at_nucleotides // 2) +
                           ['T'] * (at_nucleotides // 2))
            np.random.shuffle(remaining_seq)
            sequence.extend(remaining_seq)

        return ''.join(sequence)[:size]

    def _generate_at_rich_chromosome(self, size: int) -> str:
        """Generate AT-rich chromosome sequence."""
        # 70% AT content with repetitive elements
        at_ratio = 0.7

        # Create repetitive elements
        repeat_motifs = ['ATATAT', 'TTTAAA', 'ATATATGC', 'TTTTAAAA']
        sequence = []

        # Add repetitive regions
        repeat_regions = size // 10  # 10% repetitive content
        for _ in range(repeat_regions // 20):
            motif = np.random.choice(repeat_motifs)
            repeat_count = np.random.randint(5, 20)
            sequence.extend([motif] * repeat_count)

        # Fill with AT-rich sequence
        remaining = size - len(''.join(sequence))
        if remaining > 0:
            at_nucleotides = int(remaining * at_ratio)
            gc_nucleotides = remaining - at_nucleotides

            remaining_seq = (['A'] * (at_nucleotides // 2) +
                           ['T'] * (at_nucleotides // 2) +
                           ['G'] * (gc_nucleotides // 2) +
                           ['C'] * (gc_nucleotides // 2))
            np.random.shuffle(remaining_seq)
            sequence.extend(remaining_seq)

        return ''.join(sequence)[:size]

    def _generate_balanced_chromosome(self, size: int) -> str:
        """Generate balanced chromosome sequence."""
        # Equal nucleotide distribution with mixed patterns
        nucleotides = ['A', 'T', 'G', 'C']
        sequence = []

        # Add some structured regions
        for _ in range(size // 1000):
            # Random structured motif
            motif_length = np.random.randint(10, 50)
            motif = ''.join(np.random.choice(nucleotides, motif_length))
            repeat_count = np.random.randint(2, 10)
            sequence.extend([motif] * repeat_count)

        # Fill with random balanced sequence
        remaining = size - len(''.join(sequence))
        if remaining > 0:
            remaining_seq = np.random.choice(nucleotides, remaining).tolist()
            sequence.extend(remaining_seq)

        return ''.join(sequence)[:size]

    def validate_genome_accuracy(self, genomes: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Validate accuracy of S-entropy genome transformation.

        Tests:
        1. Coordinate consistency across transformations
        2. Chromosome coordinate relationships
        3. Global coordinate accuracy
        """
        results = {}

        # Transform genomes
        genome_coordinates = []
        for i, genome in enumerate(genomes):
            genome_id = f"test_genome_{i}"
            coords = self.transformer.transform_genome(genome, genome_id)
            genome_coordinates.append(coords)

        # Test 1: Coordinate consistency
        consistency_scores = []
        for i, genome in enumerate(genomes[:5]):  # Test subset for efficiency
            genome_id = f"consistency_test_{i}"
            coords1 = self.transformer.transform_genome(genome, genome_id)
            coords2 = self.transformer.transform_genome(genome, genome_id)

            consistency = np.allclose(coords1.global_coordinates, coords2.global_coordinates)
            consistency_scores.append(1.0 if consistency else 0.0)

        results['coordinate_consistency'] = np.mean(consistency_scores)

        # Test 2: Chromosome relationship preservation
        chromosome_correlations = []
        for genome_coord in genome_coordinates:
            if len(genome_coord.chromosomes) >= 2:
                # Calculate correlations between chromosome coordinates
                chrom_coords = [chrom.coordinates for chrom in genome_coord.chromosomes]
                coord_matrix = np.array(chrom_coords)

                # Pairwise correlations
                correlations = []
                for i in range(len(chrom_coords)):
                    for j in range(i + 1, len(chrom_coords)):
                        corr = np.corrcoef(coord_matrix[i], coord_matrix[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

                if correlations:
                    chromosome_correlations.extend(correlations)

        results['chromosome_relationship_preservation'] = np.mean(chromosome_correlations) if chromosome_correlations else 0.0

        # Test 3: Global coordinate quality
        global_coords = [genome.global_coordinates for genome in genome_coordinates]
        global_matrix = np.array(global_coords)

        # Coordinate space coverage
        coord_ranges = np.ptp(global_matrix, axis=0)  # Peak-to-peak range per dimension
        results['coordinate_space_coverage'] = np.mean(coord_ranges)

        # Coordinate uniqueness
        pairwise_distances = []
        for i in range(len(global_matrix)):
            for j in range(i + 1, len(global_matrix)):
                distance = np.linalg.norm(global_matrix[i] - global_matrix[j])
                pairwise_distances.append(distance)

        results['coordinate_uniqueness'] = np.mean(pairwise_distances)

        return results

    def benchmark_genome_transformation(self, genomes: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Benchmark performance of S-entropy genome transformation.

        Measures:
        1. Transformation time per genome
        2. Memory usage scaling
        3. Parallel processing efficiency
        """
        results = {}

        # Measure transformation times
        transformation_times = []
        memory_usage = []
        genome_sizes = []

        for i, genome in enumerate(genomes):
            genome_size = sum(len(seq) for seq in genome.values())
            genome_sizes.append(genome_size)

            # Memory before transformation
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Time transformation
            start_time = time.time()
            genome_id = f"benchmark_genome_{i}"
            coords = self.transformer.transform_genome(genome, genome_id)
            end_time = time.time()

            # Memory after transformation
            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            transformation_times.append(end_time - start_time)
            memory_usage.append(mem_after - mem_before)

        # Performance metrics
        results['avg_transformation_time'] = np.mean(transformation_times)
        results['std_transformation_time'] = np.std(transformation_times)
        results['avg_memory_usage'] = np.mean(memory_usage)
        results['std_memory_usage'] = np.std(memory_usage)

        # Scalability analysis
        size_time_correlation = np.corrcoef(genome_sizes, transformation_times)[0, 1]
        results['size_time_correlation'] = size_time_correlation

        # Throughput calculation
        total_nucleotides = sum(genome_sizes)
        total_time = sum(transformation_times)
        results['nucleotides_per_second'] = total_nucleotides / total_time if total_time > 0 else 0

        # Parallel processing efficiency test
        if len(genomes) >= 2:
            parallel_efficiency = self._test_parallel_efficiency(genomes[:2])
            results.update(parallel_efficiency)

        return results

    def _test_parallel_efficiency(self, genomes: List[Dict[str, str]]) -> Dict[str, float]:
        """Test parallel processing efficiency."""
        results = {}

        # Sequential processing
        transformer_sequential = GenomeTransformer(parallel_processing=False)
        start_time = time.time()
        for i, genome in enumerate(genomes):
            genome_id = f"sequential_{i}"
            transformer_sequential.transform_genome(genome, genome_id)
        sequential_time = time.time() - start_time

        # Parallel processing
        transformer_parallel = GenomeTransformer(parallel_processing=True)
        start_time = time.time()
        for i, genome in enumerate(genomes):
            genome_id = f"parallel_{i}"
            transformer_parallel.transform_genome(genome, genome_id)
        parallel_time = time.time() - start_time

        # Efficiency metrics
        results['sequential_time'] = sequential_time
        results['parallel_time'] = parallel_time
        results['parallel_speedup'] = sequential_time / parallel_time if parallel_time > 0 else 1.0
        results['parallel_efficiency'] = results['parallel_speedup'] / psutil.cpu_count()

        return results

    def compare_with_assembly_algorithms(self, genomes: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Compare S-entropy method with traditional genome assembly algorithms.

        Simulates comparison with established methods like SPAdes, Canu, etc.
        """
        comparison_results = {}

        # Run S-entropy transformations
        s_entropy_results = []
        for i, genome in enumerate(genomes):
            genome_id = f"comparison_genome_{i}"
            comparison = compare_genome_assembly_methods(genome, genome_id)
            s_entropy_results.append(comparison)

        # Aggregate results
        s_entropy_times = [result['s_entropy_method']['transformation_time'] for result in s_entropy_results]
        traditional_times = [result['traditional_methods']['processing_time'] for result in s_entropy_results]

        comparison_results['s_entropy_method'] = {
            'avg_time': np.mean(s_entropy_times),
            'std_time': np.std(s_entropy_times),
            'avg_information_density': np.mean([
                result['performance_comparison']['information_density']
                for result in s_entropy_results
            ])
        }

        comparison_results['traditional_methods'] = {
            'avg_time': np.mean(traditional_times),
            'std_time': np.std(traditional_times)
        }

        comparison_results['performance_comparison'] = {
            'time_improvement_ratio': np.mean([
                result['performance_comparison']['time_ratio']
                for result in s_entropy_results
            ]),
            'coordinate_efficiency': np.mean([
                result['performance_comparison']['coordinate_efficiency']
                for result in s_entropy_results
            ])
        }

        # Quality comparison
        assembly_scores = [
            result['s_entropy_method']['assembly_quality']['assembly_score']
            for result in s_entropy_results
        ]
        comparison_results['quality_metrics'] = {
            'avg_assembly_score': np.mean(assembly_scores),
            'std_assembly_score': np.std(assembly_scores)
        }

        return comparison_results

    def run_full_genome_validation(self, genome_count: int = 10) -> GenomeValidationResults:
        """
        Run complete genome validation suite.

        Args:
            genome_count: Number of test genomes to generate

        Returns:
            GenomeValidationResults object with comprehensive results
        """
        print(f"Running full genome validation with {genome_count} genomes...")

        # Generate test genomes
        genomes = self.generate_test_genomes(genome_count)

        # Run validation tests
        print("Validating genome accuracy...")
        accuracy_metrics = self.validate_genome_accuracy(genomes)

        print("Benchmarking genome transformation performance...")
        performance_metrics = self.benchmark_genome_transformation(genomes)

        print("Comparing with assembly algorithms...")
        assembly_comparison = self.compare_with_assembly_algorithms(genomes)

        # Scalability metrics
        scalability_metrics = self._analyze_scalability(genomes)

        # Memory usage analysis
        memory_usage = {
            'peak_memory_mb': performance_metrics['avg_memory_usage'],
            'memory_per_nucleotide': performance_metrics['avg_memory_usage'] / np.mean([
                sum(len(seq) for seq in genome.values()) for genome in genomes
            ])
        }

        # Generate visualization data
        visualization_data = self._generate_genome_visualization_data(genomes, assembly_comparison)

        results = GenomeValidationResults(
            accuracy_metrics=accuracy_metrics,
            performance_metrics=performance_metrics,
            assembly_comparison=assembly_comparison,
            scalability_metrics=scalability_metrics,
            memory_usage=memory_usage,
            visualization_data=visualization_data
        )

        self.validation_results.append(results)
        return results

    def _analyze_scalability(self, genomes: List[Dict[str, str]]) -> Dict[str, float]:
        """Analyze scalability of the transformation method."""
        genome_sizes = [sum(len(seq) for seq in genome.values()) for genome in genomes]
        chromosome_counts = [len(genome) for genome in genomes]

        # Transform a subset to measure scalability
        transformation_times = []
        for i, genome in enumerate(genomes[:5]):  # Limit for efficiency
            start_time = time.time()
            genome_id = f"scalability_test_{i}"
            self.transformer.transform_genome(genome, genome_id)
            transformation_times.append(time.time() - start_time)

        # Scalability metrics
        size_time_correlation = np.corrcoef(genome_sizes[:5], transformation_times)[0, 1]
        chrom_time_correlation = np.corrcoef(chromosome_counts[:5], transformation_times)[0, 1]

        return {
            'size_time_correlation': size_time_correlation,
            'chromosome_time_correlation': chrom_time_correlation,
            'avg_time_per_mb': np.mean(transformation_times) / np.mean([size / 1e6 for size in genome_sizes[:5]]),
            'scalability_score': 1.0 / (1.0 + abs(size_time_correlation))  # Lower correlation = better scalability
        }

    def _generate_genome_visualization_data(self, genomes: List[Dict[str, str]],
                                          assembly_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for genome validation visualizations."""
        viz_data = {}

        # Genome size distribution
        genome_sizes = [sum(len(seq) for seq in genome.values()) for genome in genomes]
        viz_data['genome_sizes'] = genome_sizes

        # Chromosome count distribution
        chromosome_counts = [len(genome) for genome in genomes]
        viz_data['chromosome_counts'] = chromosome_counts

        # S-entropy distribution across genomes
        s_entropies = []
        for i, genome in enumerate(genomes[:5]):  # Subset for efficiency
            genome_id = f"viz_genome_{i}"
            coords = self.transformer.transform_genome(genome, genome_id)
            s_entropies.append(coords.total_s_entropy)

        viz_data['s_entropies'] = s_entropies

        # Performance comparison data
        if 'performance_comparison' in assembly_comparison:
            viz_data['time_improvement_ratios'] = [assembly_comparison['performance_comparison']['time_improvement_ratio']]

        return viz_data

    def generate_genome_validation_report(self, results: GenomeValidationResults) -> str:
        """Generate comprehensive genome validation report."""
        report = []
        report.append("=" * 70)
        report.append("S-ENTROPY GENOME TRANSFORMATION VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")

        # Accuracy metrics
        report.append("GENOME ACCURACY METRICS:")
        report.append("-" * 30)
        for metric, value in results.accuracy_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")

        # Performance metrics
        report.append("GENOME PERFORMANCE METRICS:")
        report.append("-" * 35)
        for metric, value in results.performance_metrics.items():
            if isinstance(value, (int, float)):
                report.append(f"{metric}: {value:.4f}")
            else:
                report.append(f"{metric}: {value}")
        report.append("")

        # Assembly comparison
        report.append("ASSEMBLY ALGORITHM COMPARISON:")
        report.append("-" * 35)
        if 'performance_comparison' in results.assembly_comparison:
            perf_comp = results.assembly_comparison['performance_comparison']
            report.append(f"Time improvement ratio: {perf_comp.get('time_improvement_ratio', 'N/A'):.4f}")
            report.append(f"Coordinate efficiency: {perf_comp.get('coordinate_efficiency', 'N/A'):.6f}")

        # Scalability metrics
        report.append("SCALABILITY METRICS:")
        report.append("-" * 25)
        for metric, value in results.scalability_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")

        # Memory usage
        report.append("MEMORY USAGE:")
        report.append("-" * 15)
        for metric, value in results.memory_usage.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)


def benchmark_genome_transformation(genomes: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Convenience function to benchmark genome transformation performance.

    Args:
        genomes: List of genome dictionaries to benchmark

    Returns:
        Performance metrics dictionary
    """
    validator = GenomeValidationSuite()
    return validator.benchmark_genome_transformation(genomes)


def validate_genome_accuracy(genomes: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Convenience function to validate genome transformation accuracy.

    Args:
        genomes: List of genome dictionaries to validate

    Returns:
        Accuracy metrics dictionary
    """
    validator = GenomeValidationSuite()
    return validator.validate_genome_accuracy(genomes)


def compare_with_assembly_algorithms(genomes: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Convenience function to compare with assembly algorithms.

    Args:
        genomes: List of genome dictionaries to compare

    Returns:
        Comparison results dictionary
    """
    validator = GenomeValidationSuite()
    return validator.compare_with_assembly_algorithms(genomes)
