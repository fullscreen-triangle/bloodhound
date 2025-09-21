"""
Genome Module - S-Entropy Coordinate Transformation for Whole Genomes

This module implements the transformation of complete genomic datasets into S-entropy
coordinate representations, enabling distributed analysis without traditional storage.
"""

from .genome_transform import (
    GenomeTransformer,
    transform_genome_to_coordinates,
    GenomeCoordinates,
    ChromosomeCoordinates,
    compare_genome_assembly_methods
)

from .genome_validation import (
    GenomeValidationSuite,
    benchmark_genome_transformation,
    validate_genome_accuracy,
    compare_with_assembly_algorithms
)

__all__ = [
    "GenomeTransformer",
    "transform_genome_to_coordinates",
    "GenomeCoordinates",
    "ChromosomeCoordinates",
    "compare_genome_assembly_methods",
    "GenomeValidationSuite",
    "benchmark_genome_transformation",
    "validate_genome_accuracy",
    "compare_with_assembly_algorithms"
]
