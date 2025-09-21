"""
Sequence Module - S-Entropy Coordinate Transformation for Genomic Sequences

This module implements the transformation of genomic sequences into S-entropy coordinates
as described in the St. Stellas theoretical framework.
"""

from .s_entropy_transform import (
    SequenceTransformer,
    transform_sequence_to_coordinates,
    calculate_s_entropy,
    coordinate_navigation,
    compare_with_traditional_methods
)

from .validation import (
    SequenceValidationSuite,
    benchmark_transformation_performance,
    validate_coordinate_accuracy
)

__all__ = [
    "SequenceTransformer",
    "transform_sequence_to_coordinates",
    "calculate_s_entropy",
    "coordinate_navigation",
    "compare_with_traditional_methods",
    "SequenceValidationSuite",
    "benchmark_transformation_performance",
    "validate_coordinate_accuracy"
]
