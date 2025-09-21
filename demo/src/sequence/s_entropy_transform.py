"""
S-Entropy Coordinate Transformation Implementation

Based on the St. Stellas theoretical framework for transforming genomic sequences
into navigable coordinate representations using S-entropy calculations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import time
from Bio.Seq import Seq
from Bio.SeqUtils import GC
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hashlib


@dataclass
class SEntropyCoordinates:
    """Container for S-entropy coordinate representation of a genomic sequence."""
    coordinates: np.ndarray
    s_entropy: float
    sequence_hash: str
    transformation_metadata: Dict[str, Any]
    navigation_indices: List[int]


class SequenceTransformer:
    """
    Transforms genomic sequences into S-entropy coordinate space.

    Implements the mathematical framework from St. Stellas theory where
    sequences are mapped to coordinates that preserve informational content
    while enabling efficient navigation and comparison.
    """

    def __init__(self, coordinate_dimensions: int = 64, entropy_window: int = 21):
        self.coordinate_dimensions = coordinate_dimensions
        self.entropy_window = entropy_window
        self.nucleotide_mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}

    def calculate_s_entropy(self, sequence: str) -> float:
        """
        Calculate S-entropy for a genomic sequence.

        S-entropy represents the informational content that enables
        coordinate navigation without traditional pattern storage.

        Args:
            sequence: DNA sequence string

        Returns:
            S-entropy value for the sequence
        """
        if len(sequence) == 0:
            return 0.0

        # Convert sequence to numerical representation
        numeric_seq = [self.nucleotide_mapping.get(nuc, 4) for nuc in sequence.upper()]

        # Calculate local entropy windows
        entropies = []
        for i in range(0, len(numeric_seq) - self.entropy_window + 1, self.entropy_window // 2):
            window = numeric_seq[i:i + self.entropy_window]

            # Calculate Shannon entropy for window
            unique, counts = np.unique(window, return_counts=True)
            probabilities = counts / len(window)
            shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

            # S-entropy modification: weight by positional information
            positional_weight = np.exp(-i / len(numeric_seq))
            s_entropy_component = shannon_entropy * positional_weight
            entropies.append(s_entropy_component)

        # Global S-entropy as weighted sum
        return np.sum(entropies) / len(entropies) if entropies else 0.0

    def _extract_sequence_features(self, sequence: str) -> np.ndarray:
        """Extract comprehensive features from genomic sequence for coordinate mapping."""
        features = []

        # Basic composition features
        gc_content = GC(sequence) / 100.0
        features.append(gc_content)

        # Nucleotide frequencies
        for nuc in ['A', 'T', 'G', 'C']:
            freq = sequence.count(nuc) / len(sequence)
            features.append(freq)

        # Dinucleotide frequencies
        dinucs = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC',
                  'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        for dinuc in dinucs:
            count = sum(1 for i in range(len(sequence) - 1)
                       if sequence[i:i+2] == dinuc)
            freq = count / (len(sequence) - 1) if len(sequence) > 1 else 0
            features.append(freq)

        # Local complexity measures
        complexities = []
        window_size = min(100, len(sequence) // 10)
        if window_size > 0:
            for i in range(0, len(sequence) - window_size + 1, window_size):
                window = sequence[i:i + window_size]
                complexity = len(set(window)) / len(window)
                complexities.append(complexity)

        # Pad or truncate to fixed size
        target_complexity_features = 10
        if len(complexities) < target_complexity_features:
            complexities.extend([0.0] * (target_complexity_features - len(complexities)))
        else:
            complexities = complexities[:target_complexity_features]

        features.extend(complexities)

        return np.array(features)

    def transform_to_coordinates(self, sequence: str) -> SEntropyCoordinates:
        """
        Transform genomic sequence to S-entropy coordinates.

        Args:
            sequence: DNA sequence string

        Returns:
            SEntropyCoordinates object containing coordinate representation
        """
        start_time = time.time()

        # Calculate S-entropy
        s_entropy = self.calculate_s_entropy(sequence)

        # Extract sequence features
        features = self._extract_sequence_features(sequence)

        # Generate coordinate representation
        # Use S-entropy to seed coordinate generation
        np.random.seed(int(s_entropy * 1000000) % (2**32))

        # Create base coordinates from features
        if len(features) >= self.coordinate_dimensions:
            coordinates = features[:self.coordinate_dimensions]
        else:
            # Pad with S-entropy derived values
            padding_size = self.coordinate_dimensions - len(features)
            entropy_derived = np.random.normal(s_entropy, 0.1, padding_size)
            coordinates = np.concatenate([features, entropy_derived])

        # Apply S-entropy transformation matrix
        transformation_matrix = self._generate_transformation_matrix(s_entropy)
        coordinates = np.dot(transformation_matrix, coordinates)

        # Generate navigation indices for coordinate space navigation
        navigation_indices = self._generate_navigation_indices(coordinates, sequence)

        # Create sequence hash for verification
        sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()

        # Metadata
        metadata = {
            'sequence_length': len(sequence),
            'transformation_time': time.time() - start_time,
            'coordinate_dimensions': self.coordinate_dimensions,
            'entropy_window': self.entropy_window,
            'feature_count': len(features)
        }

        return SEntropyCoordinates(
            coordinates=coordinates,
            s_entropy=s_entropy,
            sequence_hash=sequence_hash,
            transformation_metadata=metadata,
            navigation_indices=navigation_indices
        )

    def _generate_transformation_matrix(self, s_entropy: float) -> np.ndarray:
        """Generate S-entropy dependent transformation matrix."""
        # Create deterministic but entropy-dependent transformation
        seed = int(s_entropy * 1000000) % (2**32)
        np.random.seed(seed)

        # Generate orthogonal transformation matrix
        random_matrix = np.random.randn(self.coordinate_dimensions, self.coordinate_dimensions)
        q, r = np.linalg.qr(random_matrix)

        # Scale by S-entropy to preserve informational content
        scaling = np.diag(np.random.uniform(0.5, 1.5, self.coordinate_dimensions))
        transformation_matrix = np.dot(q, scaling)

        return transformation_matrix

    def _generate_navigation_indices(self, coordinates: np.ndarray, sequence: str) -> List[int]:
        """Generate indices for efficient coordinate space navigation."""
        # Create navigation indices based on coordinate magnitudes and sequence properties
        coord_magnitudes = np.abs(coordinates)
        sorted_indices = np.argsort(coord_magnitudes)[::-1]

        # Select top navigation points
        navigation_count = min(10, len(coordinates))
        navigation_indices = sorted_indices[:navigation_count].tolist()

        return navigation_indices


def transform_sequence_to_coordinates(sequence: str,
                                    coordinate_dimensions: int = 64,
                                    entropy_window: int = 21) -> SEntropyCoordinates:
    """
    Convenience function to transform a single sequence to S-entropy coordinates.

    Args:
        sequence: DNA sequence string
        coordinate_dimensions: Number of coordinate dimensions
        entropy_window: Window size for entropy calculation

    Returns:
        SEntropyCoordinates object
    """
    transformer = SequenceTransformer(coordinate_dimensions, entropy_window)
    return transformer.transform_to_coordinates(sequence)


def calculate_s_entropy(sequence: str, entropy_window: int = 21) -> float:
    """
    Calculate S-entropy for a genomic sequence.

    Args:
        sequence: DNA sequence string
        entropy_window: Window size for entropy calculation

    Returns:
        S-entropy value
    """
    transformer = SequenceTransformer(entropy_window=entropy_window)
    return transformer.calculate_s_entropy(sequence)


def coordinate_navigation(coordinates_list: List[SEntropyCoordinates],
                         query_coordinates: SEntropyCoordinates,
                         navigation_threshold: float = 0.1) -> List[Tuple[int, float]]:
    """
    Navigate coordinate space to find similar sequences.

    Args:
        coordinates_list: List of coordinate objects to search
        query_coordinates: Query coordinate object
        navigation_threshold: Similarity threshold for navigation

    Returns:
        List of (index, distance) tuples for similar coordinates
    """
    results = []
    query_coords = query_coordinates.coordinates

    for i, coord_obj in enumerate(coordinates_list):
        # Calculate coordinate distance
        distance = np.linalg.norm(query_coords - coord_obj.coordinates)

        # Check navigation indices overlap
        query_nav_set = set(query_coordinates.navigation_indices)
        coord_nav_set = set(coord_obj.navigation_indices)
        navigation_overlap = len(query_nav_set.intersection(coord_nav_set)) / len(query_nav_set.union(coord_nav_set))

        # Combined similarity score
        similarity_score = 1.0 / (1.0 + distance) * (1.0 + navigation_overlap)

        if similarity_score > navigation_threshold:
            results.append((i, similarity_score))

    # Sort by similarity score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def compare_with_traditional_methods(sequences: List[str]) -> Dict[str, Any]:
    """
    Compare S-entropy coordinate transformation with traditional sequence analysis methods.

    Args:
        sequences: List of DNA sequences to analyze

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
    transformer = SequenceTransformer()
    s_entropy_coords = [transformer.transform_to_coordinates(seq) for seq in sequences]
    s_entropy_time = time.time() - start_time

    # Extract coordinate matrix for analysis
    coord_matrix = np.array([coord.coordinates for coord in s_entropy_coords])

    # Traditional method 1: PCA on sequence features
    start_time = time.time()
    feature_matrix = np.array([transformer._extract_sequence_features(seq) for seq in sequences])
    pca = PCA(n_components=min(64, feature_matrix.shape[1]))
    pca_coords = pca.fit_transform(feature_matrix)
    pca_time = time.time() - start_time

    # Traditional method 2: t-SNE
    start_time = time.time()
    if len(sequences) > 4:  # t-SNE requires at least 4 samples
        tsne = TSNE(n_components=2, random_state=42)
        tsne_coords = tsne.fit_transform(feature_matrix)
    else:
        tsne_coords = np.zeros((len(sequences), 2))
    tsne_time = time.time() - start_time

    # Calculate pairwise distances for comparison
    s_entropy_distances = pdist(coord_matrix)
    pca_distances = pdist(pca_coords)
    tsne_distances = pdist(tsne_coords) if len(sequences) > 4 else np.array([])

    # Store results
    results['s_entropy_method'] = {
        'coordinates': coord_matrix,
        'transformation_time': s_entropy_time,
        'pairwise_distances': s_entropy_distances,
        'average_s_entropy': np.mean([coord.s_entropy for coord in s_entropy_coords])
    }

    results['traditional_methods'] = {
        'pca': {
            'coordinates': pca_coords,
            'transformation_time': pca_time,
            'pairwise_distances': pca_distances,
            'explained_variance_ratio': pca.explained_variance_ratio_
        },
        'tsne': {
            'coordinates': tsne_coords,
            'transformation_time': tsne_time,
            'pairwise_distances': tsne_distances
        }
    }

    # Performance comparison
    results['performance_comparison'] = {
        'time_ratio_pca': s_entropy_time / pca_time if pca_time > 0 else float('inf'),
        'time_ratio_tsne': s_entropy_time / tsne_time if tsne_time > 0 else float('inf'),
        'coordinate_dimensions': {
            's_entropy': coord_matrix.shape[1],
            'pca': pca_coords.shape[1],
            'tsne': tsne_coords.shape[1]
        }
    }

    return results
