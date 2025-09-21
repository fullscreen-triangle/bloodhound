"""
Whole Genome S-Entropy Coordinate Transformation

Implements S-entropy coordinate transformation for complete genomes, including
chromosome-level analysis and comparison with established assembly algorithms.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio.Seq import Seq
from Bio.SeqUtils import GC
import pandas as pd

from ..sequence.s_entropy_transform import SequenceTransformer, SEntropyCoordinates


@dataclass
class ChromosomeCoordinates:
    """S-entropy coordinate representation of a single chromosome."""
    chromosome_id: str
    coordinates: np.ndarray
    s_entropy: float
    length: int
    gc_content: float
    gene_density: float
    repetitive_content: float
    coordinate_segments: List[SEntropyCoordinates]
    metadata: Dict[str, Any]


@dataclass
class GenomeCoordinates:
    """Complete genome S-entropy coordinate representation."""
    genome_id: str
    chromosomes: List[ChromosomeCoordinates]
    global_coordinates: np.ndarray
    total_s_entropy: float
    genome_size: int
    assembly_quality_metrics: Dict[str, float]
    coordinate_navigation_map: Dict[str, List[int]]
    transformation_metadata: Dict[str, Any]


class GenomeTransformer:
    """
    Transforms complete genomes into S-entropy coordinate representations.

    Handles chromosome-level segmentation, global coordinate integration,
    and comparison with traditional genome assembly methods.
    """

    def __init__(self,
                 segment_size: int = 10000,
                 coordinate_dimensions: int = 128,
                 parallel_processing: bool = True,
                 max_workers: Optional[int] = None):
        self.segment_size = segment_size
        self.coordinate_dimensions = coordinate_dimensions
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        self.sequence_transformer = SequenceTransformer(coordinate_dimensions)

    def transform_chromosome(self, chromosome_sequence: str, chromosome_id: str) -> ChromosomeCoordinates:
        """
        Transform a single chromosome to S-entropy coordinates.

        Args:
            chromosome_sequence: Complete chromosome DNA sequence
            chromosome_id: Identifier for the chromosome

        Returns:
            ChromosomeCoordinates object
        """
        start_time = time.time()

        # Segment chromosome into manageable pieces
        segments = self._segment_chromosome(chromosome_sequence)

        # Transform segments to coordinates
        if self.parallel_processing and len(segments) > 1:
            segment_coordinates = self._transform_segments_parallel(segments)
        else:
            segment_coordinates = [
                self.sequence_transformer.transform_to_coordinates(segment)
                for segment in segments
            ]

        # Calculate chromosome-level metrics
        gc_content = GC(chromosome_sequence)
        gene_density = self._estimate_gene_density(chromosome_sequence)
        repetitive_content = self._estimate_repetitive_content(chromosome_sequence)

        # Aggregate segment coordinates into chromosome coordinates
        chromosome_coords = self._aggregate_segment_coordinates(segment_coordinates)

        # Calculate chromosome S-entropy
        chromosome_s_entropy = np.mean([coord.s_entropy for coord in segment_coordinates])

        # Metadata
        metadata = {
            'transformation_time': time.time() - start_time,
            'segment_count': len(segments),
            'segment_size': self.segment_size,
            'coordinate_dimensions': self.coordinate_dimensions
        }

        return ChromosomeCoordinates(
            chromosome_id=chromosome_id,
            coordinates=chromosome_coords,
            s_entropy=chromosome_s_entropy,
            length=len(chromosome_sequence),
            gc_content=gc_content,
            gene_density=gene_density,
            repetitive_content=repetitive_content,
            coordinate_segments=segment_coordinates,
            metadata=metadata
        )

    def transform_genome(self,
                        genome_sequences: Dict[str, str],
                        genome_id: str) -> GenomeCoordinates:
        """
        Transform complete genome to S-entropy coordinates.

        Args:
            genome_sequences: Dictionary mapping chromosome IDs to sequences
            genome_id: Identifier for the genome

        Returns:
            GenomeCoordinates object
        """
        start_time = time.time()

        # Transform each chromosome
        chromosomes = []
        if self.parallel_processing:
            chromosomes = self._transform_chromosomes_parallel(genome_sequences)
        else:
            for chrom_id, sequence in genome_sequences.items():
                chrom_coords = self.transform_chromosome(sequence, chrom_id)
                chromosomes.append(chrom_coords)

        # Create global genome coordinates
        global_coordinates = self._create_global_coordinates(chromosomes)

        # Calculate genome-level metrics
        total_s_entropy = sum(chrom.s_entropy for chrom in chromosomes)
        genome_size = sum(chrom.length for chrom in chromosomes)

        # Assembly quality metrics
        assembly_quality = self._calculate_assembly_quality_metrics(chromosomes)

        # Create coordinate navigation map
        navigation_map = self._create_coordinate_navigation_map(chromosomes)

        # Transformation metadata
        transformation_metadata = {
            'transformation_time': time.time() - start_time,
            'chromosome_count': len(chromosomes),
            'total_segments': sum(len(chrom.coordinate_segments) for chrom in chromosomes),
            'parallel_processing': self.parallel_processing
        }

        return GenomeCoordinates(
            genome_id=genome_id,
            chromosomes=chromosomes,
            global_coordinates=global_coordinates,
            total_s_entropy=total_s_entropy,
            genome_size=genome_size,
            assembly_quality_metrics=assembly_quality,
            coordinate_navigation_map=navigation_map,
            transformation_metadata=transformation_metadata
        )

    def _segment_chromosome(self, chromosome_sequence: str) -> List[str]:
        """Segment chromosome into overlapping pieces for coordinate transformation."""
        segments = []
        overlap = self.segment_size // 4  # 25% overlap

        for i in range(0, len(chromosome_sequence), self.segment_size - overlap):
            segment = chromosome_sequence[i:i + self.segment_size]
            if len(segment) >= self.segment_size // 2:  # Only include substantial segments
                segments.append(segment)

        return segments

    def _transform_segments_parallel(self, segments: List[str]) -> List[SEntropyCoordinates]:
        """Transform chromosome segments in parallel."""
        segment_coordinates = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit transformation tasks
            future_to_segment = {
                executor.submit(self.sequence_transformer.transform_to_coordinates, segment): i
                for i, segment in enumerate(segments)
            }

            # Collect results in order
            results = [None] * len(segments)
            for future in as_completed(future_to_segment):
                segment_idx = future_to_segment[future]
                try:
                    coord = future.result()
                    results[segment_idx] = coord
                except Exception as e:
                    print(f"Segment {segment_idx} transformation failed: {e}")
                    # Create dummy coordinates for failed segments
                    results[segment_idx] = SEntropyCoordinates(
                        coordinates=np.zeros(self.coordinate_dimensions),
                        s_entropy=0.0,
                        sequence_hash="",
                        transformation_metadata={},
                        navigation_indices=[]
                    )

            segment_coordinates = [coord for coord in results if coord is not None]

        return segment_coordinates

    def _transform_chromosomes_parallel(self, genome_sequences: Dict[str, str]) -> List[ChromosomeCoordinates]:
        """Transform chromosomes in parallel."""
        chromosomes = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chromosome transformation tasks
            future_to_chrom = {
                executor.submit(self.transform_chromosome, sequence, chrom_id): chrom_id
                for chrom_id, sequence in genome_sequences.items()
            }

            # Collect results
            for future in as_completed(future_to_chrom):
                chrom_id = future_to_chrom[future]
                try:
                    chrom_coords = future.result()
                    chromosomes.append(chrom_coords)
                except Exception as e:
                    print(f"Chromosome {chrom_id} transformation failed: {e}")

        # Sort chromosomes by ID for consistency
        chromosomes.sort(key=lambda x: x.chromosome_id)
        return chromosomes

    def _aggregate_segment_coordinates(self, segment_coordinates: List[SEntropyCoordinates]) -> np.ndarray:
        """Aggregate segment coordinates into chromosome-level coordinates."""
        if not segment_coordinates:
            return np.zeros(self.coordinate_dimensions)

        # Stack segment coordinates
        coord_matrix = np.array([coord.coordinates for coord in segment_coordinates])

        # Aggregate using weighted average based on S-entropy
        weights = np.array([coord.s_entropy for coord in segment_coordinates])
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

        # Weighted average of coordinates
        chromosome_coords = np.average(coord_matrix, axis=0, weights=weights)

        return chromosome_coords

    def _create_global_coordinates(self, chromosomes: List[ChromosomeCoordinates]) -> np.ndarray:
        """Create global genome coordinates from chromosome coordinates."""
        if not chromosomes:
            return np.zeros(self.coordinate_dimensions)

        # Stack chromosome coordinates
        chrom_coord_matrix = np.array([chrom.coordinates for chrom in chromosomes])

        # Weight by chromosome size and S-entropy
        weights = np.array([chrom.length * chrom.s_entropy for chrom in chromosomes])
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

        # Weighted average for global coordinates
        global_coords = np.average(chrom_coord_matrix, axis=0, weights=weights)

        return global_coords

    def _estimate_gene_density(self, sequence: str) -> float:
        """Estimate gene density based on sequence composition patterns."""
        # Simple heuristic: look for patterns typical of coding regions

        # Count potential start/stop codons
        start_codons = ['ATG']
        stop_codons = ['TAA', 'TAG', 'TGA']

        start_count = sum(sequence.count(codon) for codon in start_codons)
        stop_count = sum(sequence.count(codon) for codon in stop_codons)

        # Estimate based on codon frequency
        codon_density = (start_count + stop_count) / (len(sequence) / 3) if len(sequence) > 0 else 0

        # Normalize to 0-1 range (rough estimate)
        gene_density = min(1.0, codon_density * 10)  # Scaling factor based on typical gene densities

        return gene_density

    def _estimate_repetitive_content(self, sequence: str) -> float:
        """Estimate repetitive content in the sequence."""
        # Simple approach: look for repeated k-mers
        k = 6  # 6-mer repeats
        kmer_counts = {}

        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1

        # Calculate repetitive content
        total_kmers = len(sequence) - k + 1
        repetitive_kmers = sum(count - 1 for count in kmer_counts.values() if count > 1)

        repetitive_fraction = repetitive_kmers / total_kmers if total_kmers > 0 else 0
        return repetitive_fraction

    def _calculate_assembly_quality_metrics(self, chromosomes: List[ChromosomeCoordinates]) -> Dict[str, float]:
        """Calculate assembly quality metrics from chromosome coordinates."""
        metrics = {}

        if not chromosomes:
            return metrics

        # Coordinate consistency across chromosomes
        coord_matrix = np.array([chrom.coordinates for chrom in chromosomes])
        coord_std = np.std(coord_matrix, axis=0)
        metrics['coordinate_consistency'] = 1.0 / (1.0 + np.mean(coord_std))

        # S-entropy distribution
        s_entropies = [chrom.s_entropy for chrom in chromosomes]
        metrics['s_entropy_mean'] = np.mean(s_entropies)
        metrics['s_entropy_std'] = np.std(s_entropies)

        # Size distribution
        sizes = [chrom.length for chrom in chromosomes]
        metrics['size_coefficient_variation'] = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0

        # GC content variation
        gc_contents = [chrom.gc_content for chrom in chromosomes]
        metrics['gc_content_variation'] = np.std(gc_contents)

        # Overall assembly score (higher is better)
        metrics['assembly_score'] = (
            metrics['coordinate_consistency'] *
            (1.0 / (1.0 + metrics['s_entropy_std'])) *
            (1.0 / (1.0 + metrics['size_coefficient_variation']))
        )

        return metrics

    def _create_coordinate_navigation_map(self, chromosomes: List[ChromosomeCoordinates]) -> Dict[str, List[int]]:
        """Create navigation map for efficient coordinate space traversal."""
        navigation_map = {}

        for chrom in chromosomes:
            # Extract navigation indices from chromosome segments
            all_nav_indices = []
            for segment in chrom.coordinate_segments:
                all_nav_indices.extend(segment.navigation_indices)

            # Remove duplicates and sort
            unique_indices = sorted(list(set(all_nav_indices)))
            navigation_map[chrom.chromosome_id] = unique_indices

        # Global navigation indices
        all_indices = []
        for indices in navigation_map.values():
            all_indices.extend(indices)

        global_indices = sorted(list(set(all_indices)))
        navigation_map['global'] = global_indices

        return navigation_map


def transform_genome_to_coordinates(genome_sequences: Dict[str, str],
                                  genome_id: str,
                                  segment_size: int = 10000,
                                  coordinate_dimensions: int = 128,
                                  parallel_processing: bool = True) -> GenomeCoordinates:
    """
    Convenience function to transform a genome to S-entropy coordinates.

    Args:
        genome_sequences: Dictionary mapping chromosome IDs to sequences
        genome_id: Identifier for the genome
        segment_size: Size of chromosome segments for processing
        coordinate_dimensions: Number of coordinate dimensions
        parallel_processing: Whether to use parallel processing

    Returns:
        GenomeCoordinates object
    """
    transformer = GenomeTransformer(
        segment_size=segment_size,
        coordinate_dimensions=coordinate_dimensions,
        parallel_processing=parallel_processing
    )
    return transformer.transform_genome(genome_sequences, genome_id)


def compare_genome_assembly_methods(genome_sequences: Dict[str, str],
                                  genome_id: str) -> Dict[str, Any]:
    """
    Compare S-entropy genome transformation with traditional assembly methods.

    Args:
        genome_sequences: Dictionary mapping chromosome IDs to sequences
        genome_id: Identifier for the genome

    Returns:
        Comparison results dictionary
    """
    results = {
        's_entropy_method': {},
        'traditional_methods': {},
        'performance_comparison': {}
    }

    # S-entropy transformation
    start_time = time.time()
    transformer = GenomeTransformer()
    s_entropy_genome = transformer.transform_genome(genome_sequences, genome_id)
    s_entropy_time = time.time() - start_time

    # Traditional method simulation (assembly metrics)
    start_time = time.time()
    traditional_metrics = _simulate_traditional_assembly_metrics(genome_sequences)
    traditional_time = time.time() - start_time

    # Store results
    results['s_entropy_method'] = {
        'transformation_time': s_entropy_time,
        'genome_size': s_entropy_genome.genome_size,
        'total_s_entropy': s_entropy_genome.total_s_entropy,
        'assembly_quality': s_entropy_genome.assembly_quality_metrics,
        'coordinate_dimensions': len(s_entropy_genome.global_coordinates)
    }

    results['traditional_methods'] = {
        'processing_time': traditional_time,
        'assembly_metrics': traditional_metrics
    }

    results['performance_comparison'] = {
        'time_ratio': s_entropy_time / traditional_time if traditional_time > 0 else float('inf'),
        'information_density': s_entropy_genome.total_s_entropy / s_entropy_genome.genome_size,
        'coordinate_efficiency': len(s_entropy_genome.global_coordinates) / s_entropy_genome.genome_size
    }

    return results


def _simulate_traditional_assembly_metrics(genome_sequences: Dict[str, str]) -> Dict[str, float]:
    """Simulate traditional genome assembly quality metrics for comparison."""
    metrics = {}

    # Calculate basic assembly statistics
    total_length = sum(len(seq) for seq in genome_sequences.values())
    chromosome_count = len(genome_sequences)

    # N50 calculation (simplified)
    lengths = sorted([len(seq) for seq in genome_sequences.values()], reverse=True)
    cumulative_length = 0
    n50 = 0
    for length in lengths:
        cumulative_length += length
        if cumulative_length >= total_length / 2:
            n50 = length
            break

    # Assembly metrics
    metrics['total_length'] = total_length
    metrics['chromosome_count'] = chromosome_count
    metrics['n50'] = n50
    metrics['largest_chromosome'] = max(lengths) if lengths else 0
    metrics['average_chromosome_size'] = total_length / chromosome_count if chromosome_count > 0 else 0

    # Quality scores (simulated)
    metrics['contiguity_score'] = n50 / total_length if total_length > 0 else 0
    metrics['completeness_score'] = min(1.0, chromosome_count / 23)  # Assuming human genome reference

    return metrics
