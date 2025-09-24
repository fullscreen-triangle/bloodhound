"""
Distributed Mass Spectrometry Module

This module implements distributed mass spectrometry processing using S-entropy
coordinate transformation, gas molecular equilibrium, and empty dictionary synthesis.

Based on the S-entropy molecular language framework and spectrometry system.
"""

from .molecular_coordinates import (
    MolecularCoordinateTransformer,
    transform_spectrum_to_coordinates,
    transform_molecular_structure_to_coordinates,
    SEntropyMolecularCoordinates
)

from .distributed_ms_network import (
    DistributedMSNetwork,
    MSProcessingNode,
    SpectralProcessingSession,
    create_distributed_ms_network,
    analyze_distributed_spectrum
)

from .empty_dictionary_synthesis import (
    EmptyDictionaryMSSystem,
    MolecularIdentificationSynthesis,
    synthesize_molecular_identity
)

from .senn_spectrometry import (
    SEntropyNeuralNetwork,
    VarianceMinimizingProcessor,
    MiraculousCircuitProcessor,
    process_with_senn
)

from .bmd_validation import (
    BiologicalMaxwellDemonValidator,
    CrossModalPathwayValidator,
    validate_bmd_equivalence
)

from .mufakose_metabolomics import (
    MufakoseMetabolomicsFramework,
    OscillatoryMolecularAnalyzer,
    EnvironmentalComplexityOptimizer,
    OscillatorySignature,
    EnvironmentalComplexityState,
    MetabolomicConfirmation,
    analyze_metabolomic_sample,
    optimize_environmental_complexity,
    generate_oscillatory_signature
)

__all__ = [
    # Molecular coordinate transformation
    "MolecularCoordinateTransformer",
    "transform_spectrum_to_coordinates",
    "transform_molecular_structure_to_coordinates",
    "SEntropyMolecularCoordinates",

    # Distributed MS network
    "DistributedMSNetwork",
    "MSProcessingNode",
    "SpectralProcessingSession",
    "create_distributed_ms_network",
    "analyze_distributed_spectrum",

    # Empty dictionary synthesis
    "EmptyDictionaryMSSystem",
    "MolecularIdentificationSynthesis",
    "synthesize_molecular_identity",

    # SENN processing
    "SEntropyNeuralNetwork",
    "VarianceMinimizingProcessor",
    "MiraculousCircuitProcessor",
    "process_with_senn",

    # BMD validation
    "BiologicalMaxwellDemonValidator",
    "CrossModalPathwayValidator",
    "validate_bmd_equivalence",

    # Mufakose Metabolomics Framework
    "MufakoseMetabolomicsFramework",
    "OscillatoryMolecularAnalyzer",
    "EnvironmentalComplexityOptimizer",
    "OscillatorySignature",
    "EnvironmentalComplexityState",
    "MetabolomicConfirmation",
    "analyze_metabolomic_sample",
    "optimize_environmental_complexity",
    "generate_oscillatory_signature"
]
