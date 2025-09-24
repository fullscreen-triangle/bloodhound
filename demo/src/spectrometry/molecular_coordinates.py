"""
S-Entropy Molecular Coordinate Transformation

Implements the mathematical framework for transforming mass spectrometry data
and molecular structures into S-entropy coordinate space for distributed processing.

Based on: "S-Entropy Molecular Coordinate Transformation: Mathematical Framework
for Raw Data Conversion to Multi-Dimensional Entropy Space"
"""

import numpy as np
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import re
from collections import Counter


@dataclass
class SEntropyMolecularCoordinates:
    """Represents molecular data in S-entropy coordinate space."""
    coordinates: np.ndarray  # (S_knowledge, S_time, S_entropy)
    molecular_type: str  # 'spectrum', 'structure', 'genomic', 'protein'
    original_data: Dict[str, Any]
    transformation_metadata: Dict[str, Any] = field(default_factory=dict)
    coordinate_path: Optional[np.ndarray] = None  # Full path for sequences


class MolecularCoordinateTransformer:
    """
    Transforms molecular data into S-entropy coordinate space.

    Supports:
    - Mass spectrometry data (peaks, intensities, molecular ions)
    - Chemical structures (SMILES, molecular descriptors)
    - Cross-modal validation with genomic data
    """

    def __init__(self, coordinate_dimensions: int = 3):
        self.coordinate_dimensions = coordinate_dimensions
        self.transformation_cache = {}

        # Physicochemical property databases (simplified for demo)
        self.electronegativity_values = {
            'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
            'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0.00, 'Na': 0.93, 'Mg': 1.31,
            'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0.00
        }

        self.atomic_weights = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 'C': 12.011,
            'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
            'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948
        }

    def transform_spectrum_to_coordinates(self,
                                        spectrum_data: Dict[str, Any],
                                        context_window: int = 10) -> SEntropyMolecularCoordinates:
        """
        Transform mass spectrometry data to S-entropy coordinates.

        Args:
            spectrum_data: Dict containing 'mz_values', 'intensities', 'metadata'
            context_window: Window size for sliding window analysis

        Returns:
            SEntropyMolecularCoordinates object
        """
        mz_values = spectrum_data.get('mz_values', [])
        intensities = spectrum_data.get('intensities', [])
        metadata = spectrum_data.get('metadata', {})

        if len(mz_values) != len(intensities):
            raise ValueError("m/z values and intensities must have same length")

        # Convert spectrum to coordinate path
        coordinate_path = []

        for i, (mz, intensity) in enumerate(zip(mz_values, intensities)):
            # Define sliding window
            window_start = max(0, i - context_window // 2)
            window_end = min(len(mz_values), i + context_window // 2 + 1)

            window_mz = mz_values[window_start:window_end]
            window_intensities = intensities[window_start:window_end]

            # Calculate S-entropy coordinates
            s_knowledge = self._calculate_spectral_knowledge_coordinate(
                mz, intensity, window_mz, window_intensities
            )
            s_time = self._calculate_spectral_time_coordinate(
                i, len(mz_values), window_mz, window_intensities
            )
            s_entropy = self._calculate_spectral_entropy_coordinate(
                window_mz, window_intensities
            )

            coordinate_path.append([s_knowledge, s_time, s_entropy])

        coordinate_path = np.array(coordinate_path)

        # Calculate final coordinates as path integration
        final_coordinates = np.sum(coordinate_path, axis=0)

        # Normalize coordinates
        final_coordinates = self._normalize_coordinates(final_coordinates)

        transformation_metadata = {
            'spectrum_length': len(mz_values),
            'mz_range': [min(mz_values), max(mz_values)] if mz_values else [0, 0],
            'intensity_range': [min(intensities), max(intensities)] if intensities else [0, 0],
            'context_window': context_window,
            'transformation_time': time.time()
        }

        return SEntropyMolecularCoordinates(
            coordinates=final_coordinates,
            molecular_type='spectrum',
            original_data=spectrum_data,
            transformation_metadata=transformation_metadata,
            coordinate_path=coordinate_path
        )

    def transform_molecular_structure_to_coordinates(self,
                                                   structure_data: Dict[str, Any]) -> SEntropyMolecularCoordinates:
        """
        Transform chemical structure data to S-entropy coordinates.

        Args:
            structure_data: Dict containing 'smiles', 'molecular_formula', 'descriptors'

        Returns:
            SEntropyMolecularCoordinates object
        """
        smiles = structure_data.get('smiles', '')
        molecular_formula = structure_data.get('molecular_formula', '')
        descriptors = structure_data.get('descriptors', {})

        if not smiles and not molecular_formula:
            raise ValueError("Either SMILES or molecular formula must be provided")

        # Extract functional groups from SMILES
        functional_groups = self._extract_functional_groups(smiles) if smiles else []

        # Parse molecular formula for atomic composition
        atomic_composition = self._parse_molecular_formula(molecular_formula) if molecular_formula else {}

        # Calculate S-entropy coordinates
        s_knowledge = self._calculate_structural_knowledge_coordinate(
            functional_groups, atomic_composition, descriptors
        )
        s_time = self._calculate_structural_time_coordinate(
            functional_groups, atomic_composition
        )
        s_entropy = self._calculate_structural_entropy_coordinate(
            functional_groups, atomic_composition
        )

        coordinates = np.array([s_knowledge, s_time, s_entropy])
        coordinates = self._normalize_coordinates(coordinates)

        transformation_metadata = {
            'functional_groups': functional_groups,
            'atomic_composition': atomic_composition,
            'molecular_weight': self._calculate_molecular_weight(atomic_composition),
            'transformation_time': time.time()
        }

        return SEntropyMolecularCoordinates(
            coordinates=coordinates,
            molecular_type='structure',
            original_data=structure_data,
            transformation_metadata=transformation_metadata
        )

    def _calculate_spectral_knowledge_coordinate(self,
                                               mz: float,
                                               intensity: float,
                                               window_mz: List[float],
                                               window_intensities: List[float]) -> float:
        """Calculate knowledge coordinate for spectral data."""
        if not window_intensities:
            return 0.0

        # Information content based on peak significance and context
        total_intensity = sum(window_intensities)
        if total_intensity == 0:
            return 0.0

        # Relative intensity as information measure
        relative_intensity = intensity / total_intensity

        # Shannon information content
        if relative_intensity > 0:
            information_content = -relative_intensity * np.log2(relative_intensity)
        else:
            information_content = 0.0

        # Context-weighted information (how unique is this peak?)
        uniqueness = 1.0 / (1.0 + len([i for i in window_intensities if abs(i - intensity) < 0.1 * intensity]))

        return information_content * uniqueness

    def _calculate_spectral_time_coordinate(self,
                                          position: int,
                                          total_length: int,
                                          window_mz: List[float],
                                          window_intensities: List[float]) -> float:
        """Calculate time coordinate for spectral data."""
        # Sequential dynamics based on position and local trends
        positional_weight = position / total_length if total_length > 0 else 0.0

        # Local trend analysis
        if len(window_intensities) > 2:
            # Simple trend: increasing/decreasing intensities
            trend = 0.0
            for i in range(1, len(window_intensities)):
                if window_intensities[i] > window_intensities[i-1]:
                    trend += 1
                elif window_intensities[i] < window_intensities[i-1]:
                    trend -= 1

            trend_normalized = trend / (len(window_intensities) - 1) if len(window_intensities) > 1 else 0.0
        else:
            trend_normalized = 0.0

        return positional_weight + 0.1 * trend_normalized

    def _calculate_spectral_entropy_coordinate(self,
                                             window_mz: List[float],
                                             window_intensities: List[float]) -> float:
        """Calculate entropy coordinate for spectral data."""
        if not window_intensities or sum(window_intensities) == 0:
            return 0.0

        # Normalize intensities to probabilities
        total_intensity = sum(window_intensities)
        probabilities = [i / total_intensity for i in window_intensities]

        # Shannon entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalize entropy by maximum possible (log2(n))
        max_entropy = np.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy

    def _calculate_structural_knowledge_coordinate(self,
                                                 functional_groups: List[str],
                                                 atomic_composition: Dict[str, int],
                                                 descriptors: Dict[str, float]) -> float:
        """Calculate knowledge coordinate for structural data."""
        # Information content from functional groups
        group_info = len(set(functional_groups)) / max(1, len(functional_groups))

        # Information content from atomic diversity
        total_atoms = sum(atomic_composition.values())
        if total_atoms > 0:
            atom_probs = [count / total_atoms for count in atomic_composition.values()]
            atom_entropy = -sum(p * np.log2(p) for p in atom_probs if p > 0)
            max_atom_entropy = np.log2(len(atomic_composition)) if len(atomic_composition) > 1 else 1.0
            atom_info = atom_entropy / max_atom_entropy if max_atom_entropy > 0 else 0.0
        else:
            atom_info = 0.0

        # Descriptor-based information
        descriptor_info = len(descriptors) / 100.0  # Normalize by typical descriptor count

        return (group_info + atom_info + descriptor_info) / 3.0

    def _calculate_structural_time_coordinate(self,
                                            functional_groups: List[str],
                                            atomic_composition: Dict[str, int]) -> float:
        """Calculate time coordinate for structural data."""
        # Reactivity-based temporal coordinate
        reactive_groups = ['OH', 'NH2', 'COOH', 'CHO', 'C=O', 'C=C', 'aromatic']
        reactivity_score = sum(1 for group in functional_groups if group in reactive_groups)
        reactivity_normalized = reactivity_score / max(1, len(functional_groups))

        # Electronegativity-based temporal dynamics
        total_electronegativity = 0.0
        total_atoms = 0
        for atom, count in atomic_composition.items():
            if atom in self.electronegativity_values:
                total_electronegativity += self.electronegativity_values[atom] * count
                total_atoms += count

        avg_electronegativity = total_electronegativity / total_atoms if total_atoms > 0 else 0.0
        electronegativity_normalized = avg_electronegativity / 4.0  # Max electronegativity ~4.0

        return (reactivity_normalized + electronegativity_normalized) / 2.0

    def _calculate_structural_entropy_coordinate(self,
                                               functional_groups: List[str],
                                               atomic_composition: Dict[str, int]) -> float:
        """Calculate entropy coordinate for structural data."""
        # Structural disorder measure
        total_groups = len(functional_groups)
        if total_groups == 0:
            return 0.0

        # Functional group diversity entropy
        group_counts = Counter(functional_groups)
        group_probs = [count / total_groups for count in group_counts.values()]
        group_entropy = -sum(p * np.log2(p) for p in group_probs if p > 0)
        max_group_entropy = np.log2(len(group_counts)) if len(group_counts) > 1 else 1.0
        group_entropy_normalized = group_entropy / max_group_entropy if max_group_entropy > 0 else 0.0

        # Atomic composition entropy
        total_atoms = sum(atomic_composition.values())
        if total_atoms > 0:
            atom_probs = [count / total_atoms for count in atomic_composition.values()]
            atom_entropy = -sum(p * np.log2(p) for p in atom_probs if p > 0)
            max_atom_entropy = np.log2(len(atomic_composition)) if len(atomic_composition) > 1 else 1.0
            atom_entropy_normalized = atom_entropy / max_atom_entropy if max_atom_entropy > 0 else 0.0
        else:
            atom_entropy_normalized = 0.0

        return (group_entropy_normalized + atom_entropy_normalized) / 2.0

    def _extract_functional_groups(self, smiles: str) -> List[str]:
        """Extract functional groups from SMILES notation."""
        functional_groups = []

        # Common functional group patterns
        patterns = {
            'OH': r'O[H]',
            'NH2': r'N\([H]\)[H]|N[H][H]',
            'COOH': r'C\(=O\)O[H]',
            'CHO': r'C=O',
            'C=O': r'C=O',
            'C=C': r'C=C',
            'aromatic': r'c',  # Lowercase c indicates aromatic carbon
            'ether': r'O',
            'ester': r'C\(=O\)O',
            'amide': r'C\(=O\)N',
            'nitrile': r'C#N',
            'alkyne': r'C#C'
        }

        for group_name, pattern in patterns.items():
            matches = re.findall(pattern, smiles, re.IGNORECASE)
            functional_groups.extend([group_name] * len(matches))

        return functional_groups

    def _parse_molecular_formula(self, formula: str) -> Dict[str, int]:
        """Parse molecular formula into atomic composition."""
        # Simple regex to extract elements and counts
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)

        composition = {}
        for element, count_str in matches:
            count = int(count_str) if count_str else 1
            composition[element] = composition.get(element, 0) + count

        return composition

    def _calculate_molecular_weight(self, atomic_composition: Dict[str, int]) -> float:
        """Calculate molecular weight from atomic composition."""
        weight = 0.0
        for element, count in atomic_composition.items():
            if element in self.atomic_weights:
                weight += self.atomic_weights[element] * count
        return weight

    def _normalize_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Normalize coordinates to unit sphere."""
        norm = np.linalg.norm(coordinates)
        if norm > 0:
            return coordinates / norm
        return coordinates

    def cross_modal_distance(self,
                           coords1: SEntropyMolecularCoordinates,
                           coords2: SEntropyMolecularCoordinates) -> float:
        """
        Calculate cross-modal distance between molecular representations.

        Implements the cross-modal coordinate distance from the theoretical framework.
        """
        # Euclidean distance in S-entropy space
        distance = np.linalg.norm(coords1.coordinates - coords2.coordinates)

        # Weight by molecular type compatibility
        type_compatibility = self._get_type_compatibility(coords1.molecular_type, coords2.molecular_type)

        return distance / type_compatibility

    def _get_type_compatibility(self, type1: str, type2: str) -> float:
        """Get compatibility factor between molecular types."""
        compatibility_matrix = {
            ('spectrum', 'structure'): 0.8,
            ('spectrum', 'genomic'): 0.6,
            ('spectrum', 'protein'): 0.7,
            ('structure', 'genomic'): 0.5,
            ('structure', 'protein'): 0.9,
            ('genomic', 'protein'): 0.95
        }

        key = tuple(sorted([type1, type2]))
        return compatibility_matrix.get(key, 1.0)


# Convenience functions
def transform_spectrum_to_coordinates(spectrum_data: Dict[str, Any],
                                    context_window: int = 10) -> SEntropyMolecularCoordinates:
    """Convenience function for spectrum transformation."""
    transformer = MolecularCoordinateTransformer()
    return transformer.transform_spectrum_to_coordinates(spectrum_data, context_window)


def transform_molecular_structure_to_coordinates(structure_data: Dict[str, Any]) -> SEntropyMolecularCoordinates:
    """Convenience function for molecular structure transformation."""
    transformer = MolecularCoordinateTransformer()
    return transformer.transform_molecular_structure_to_coordinates(structure_data)


def calculate_cross_modal_distance(coords1: SEntropyMolecularCoordinates,
                                 coords2: SEntropyMolecularCoordinates) -> float:
    """Convenience function for cross-modal distance calculation."""
    transformer = MolecularCoordinateTransformer()
    return transformer.cross_modal_distance(coords1, coords2)
