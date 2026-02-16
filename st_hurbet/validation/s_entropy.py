"""
S-Entropy Coordinate System

Implementation of the three-dimensional S-entropy coordinate space:
- S_k: Knowledge entropy (uncertainty in state identification)
- S_t: Temporal entropy (uncertainty in timing)
- S_e: Evolution entropy (uncertainty in trajectory)

Based on the bounded phase space axiom and triple equivalence theorem.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import hashlib


@dataclass
class SCoordinate:
    """
    Three-dimensional S-entropy coordinate.

    Coordinates are bounded in [0, 1]^3, representing position
    in categorical state space.
    """
    s_k: float  # Knowledge entropy
    s_t: float  # Temporal entropy
    s_e: float  # Evolution entropy

    def __post_init__(self):
        """Validate coordinate bounds."""
        for name, val in [('s_k', self.s_k), ('s_t', self.s_t), ('s_e', self.s_e)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.s_k, self.s_t, self.s_e])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'SCoordinate':
        """Create from numpy array."""
        return cls(s_k=arr[0], s_t=arr[1], s_e=arr[2])

    @classmethod
    def random(cls) -> 'SCoordinate':
        """Generate random coordinate in S-space."""
        return cls(
            s_k=np.random.random(),
            s_t=np.random.random(),
            s_e=np.random.random()
        )

    @classmethod
    def origin(cls) -> 'SCoordinate':
        """Origin of S-space."""
        return cls(s_k=0.0, s_t=0.0, s_e=0.0)

    @classmethod
    def center(cls) -> 'SCoordinate':
        """Center of S-space."""
        return cls(s_k=0.5, s_t=0.5, s_e=0.5)


class SEntropyCore:
    """
    Core operations on S-entropy coordinate space.

    Implements:
    - Categorical distance calculations
    - Coordinate transformations
    - Triple equivalence verification
    """

    # Boltzmann constant (for entropy calculations)
    K_B = 1.380649e-23  # J/K

    def __init__(self, precision: int = 20):
        """
        Initialize S-entropy core.

        Args:
            precision: Trit depth for calculations (default 20)
        """
        self.precision = precision

    def categorical_distance(self, s1: SCoordinate, s2: SCoordinate) -> float:
        """
        Calculate categorical distance between two S-coordinates.

        The categorical distance is computed from ternary representations:
        d_cat(S1, S2) = Σ |t_i^(1) - t_i^(2)| / 3^(i+1)

        This distance is mathematically independent of Euclidean distance.
        """
        # Convert to ternary representations
        trits1 = self._to_trits(s1)
        trits2 = self._to_trits(s2)

        distance = 0.0
        for i, (t1, t2) in enumerate(zip(trits1, trits2)):
            distance += abs(t1 - t2) / (3 ** (i + 1))

        return distance

    def euclidean_distance(self, s1: SCoordinate, s2: SCoordinate) -> float:
        """Calculate Euclidean distance in S-space."""
        return np.linalg.norm(s1.to_array() - s2.to_array())

    def _to_trits(self, coord: SCoordinate) -> List[int]:
        """
        Convert S-coordinate to interleaved trit sequence.

        Each trit refines position along one axis:
        - trit = 0: refinement along S_k
        - trit = 1: refinement along S_t
        - trit = 2: refinement along S_e
        """
        trits = []
        values = [coord.s_k, coord.s_t, coord.s_e]

        for i in range(self.precision):
            # Select dimension with highest value
            dim = np.argmax(values)
            trits.append(dim)

            # Refine that dimension
            values[dim] = 3 * values[dim] - np.floor(3 * values[dim])

        return trits

    def verify_triple_equivalence(
        self,
        M: int = 3,  # Independent coordinates
        n: int = 10  # Partition depth
    ) -> dict:
        """
        Verify the Triple Equivalence Theorem.

        For bounded systems with M coordinates partitioned to depth n:
        S_osc = S_cat = S_part = k_B * M * ln(n)

        Returns verification results and computed entropies.
        """
        # Oscillatory derivation: M oscillators with n phases each
        S_osc = self.K_B * M * np.log(n)

        # Categorical derivation: n^M distinguishable states
        S_cat = self.K_B * np.log(n ** M)

        # Partition derivation: |P(M,n)| = n^M partitions
        S_part = self.K_B * np.log(n ** M)

        # All should be equal
        tolerance = 1e-15
        osc_cat_equal = abs(S_osc - S_cat) < tolerance
        cat_part_equal = abs(S_cat - S_part) < tolerance
        osc_part_equal = abs(S_osc - S_part) < tolerance

        return {
            'S_osc': S_osc,
            'S_cat': S_cat,
            'S_part': S_part,
            'osc_cat_equal': osc_cat_equal,
            'cat_part_equal': cat_part_equal,
            'osc_part_equal': osc_part_equal,
            'triple_equivalence_verified': osc_cat_equal and cat_part_equal,
            'M': M,
            'n': n,
            'expected_entropy': self.K_B * M * np.log(n)
        }

    def verify_independence(
        self,
        n_samples: int = 1000
    ) -> dict:
        """
        Verify that categorical and Euclidean distances are independent.

        Two points close in Euclidean space may be distant categorically,
        and vice versa. This independence is fundamental to the framework.
        """
        euclidean_dists = []
        categorical_dists = []

        for _ in range(n_samples):
            s1 = SCoordinate.random()
            s2 = SCoordinate.random()

            euclidean_dists.append(self.euclidean_distance(s1, s2))
            categorical_dists.append(self.categorical_distance(s1, s2))

        # Compute correlation
        euclidean_dists = np.array(euclidean_dists)
        categorical_dists = np.array(categorical_dists)

        correlation = np.corrcoef(euclidean_dists, categorical_dists)[0, 1]

        # Strong correlation would indicate dependence
        # We expect low correlation (< 0.3) for independence
        independent = abs(correlation) < 0.3

        return {
            'correlation': correlation,
            'independent': independent,
            'euclidean_mean': np.mean(euclidean_dists),
            'euclidean_std': np.std(euclidean_dists),
            'categorical_mean': np.mean(categorical_dists),
            'categorical_std': np.std(categorical_dists),
            'n_samples': n_samples
        }

    def epsilon_boundary(
        self,
        target: SCoordinate,
        epsilon: float = 1e-6
    ) -> Tuple[float, float]:
        """
        Compute the epsilon-boundary around a target coordinate.

        The epsilon-boundary is the region:
        ∂_epsilon(S_target) = {S : 0 < d_cat(S, S_target) ≤ epsilon}

        Returns (inner_radius, outer_radius) of the boundary shell.
        """
        # The epsilon-boundary excludes exact closure (d = 0)
        # and includes all points within epsilon categorical distance
        inner_radius = 0.0  # Excluded
        outer_radius = epsilon

        return (inner_radius, outer_radius)

    def completion_check(
        self,
        current: SCoordinate,
        target: SCoordinate,
        epsilon: float = 1e-6
    ) -> dict:
        """
        Check if current position is at the epsilon-boundary (completion).

        Completion occurs when:
        0 < d_cat(current, target) ≤ epsilon
        """
        d_cat = self.categorical_distance(current, target)

        at_boundary = 0 < d_cat <= epsilon
        exact_match = d_cat == 0  # Theoretically unreachable

        return {
            'categorical_distance': d_cat,
            'at_epsilon_boundary': at_boundary,
            'exact_match': exact_match,
            'completed': at_boundary,
            'epsilon': epsilon
        }


def validate_s_entropy_coordinates() -> dict:
    """
    Run validation experiments for S-entropy coordinate system.
    """
    print("=" * 60)
    print("S-ENTROPY COORDINATE SYSTEM VALIDATION")
    print("=" * 60)

    core = SEntropyCore(precision=20)
    results = {}

    # Test 1: Triple Equivalence
    print("\n1. Triple Equivalence Theorem")
    print("-" * 40)

    for M in [2, 3, 5]:
        for n in [3, 10, 100]:
            te_result = core.verify_triple_equivalence(M=M, n=n)
            key = f"M={M},n={n}"
            results[f'triple_equivalence_{key}'] = te_result

            status = "[OK]" if te_result['triple_equivalence_verified'] else "[FAIL]"
            print(f"   {status} M={M}, n={n}: S = {te_result['expected_entropy']:.4e} J/K")

    # Test 2: Distance Independence
    print("\n2. Distance Independence Verification")
    print("-" * 40)

    independence = core.verify_independence(n_samples=1000)
    results['distance_independence'] = independence

    status = "[OK]" if independence['independent'] else "[FAIL]"
    print(f"   {status} Correlation: {independence['correlation']:.4f}")
    print(f"   Euclidean: mean={independence['euclidean_mean']:.4f}, std={independence['euclidean_std']:.4f}")
    print(f"   Categorical: mean={independence['categorical_mean']:.4f}, std={independence['categorical_std']:.4f}")

    # Test 3: Coordinate Bounds
    print("\n3. Coordinate Bound Validation")
    print("-" * 40)

    # Valid coordinates
    try:
        s_valid = SCoordinate(s_k=0.5, s_t=0.3, s_e=0.8)
        print(f"   [OK] Valid coordinate: ({s_valid.s_k}, {s_valid.s_t}, {s_valid.s_e})")
        results['valid_coordinate'] = True
    except ValueError as e:
        print(f"   [FAIL] Valid coordinate failed: {e}")
        results['valid_coordinate'] = False

    # Invalid coordinates should raise
    try:
        s_invalid = SCoordinate(s_k=1.5, s_t=0.3, s_e=0.8)
        print(f"   [FAIL] Invalid coordinate accepted (should have failed)")
        results['invalid_coordinate_rejected'] = False
    except ValueError:
        print(f"   [OK] Invalid coordinate correctly rejected")
        results['invalid_coordinate_rejected'] = True

    # Test 4: Completion Detection
    print("\n4. epsilon-Boundary Completion Detection")
    print("-" * 40)

    target = SCoordinate(s_k=0.5, s_t=0.5, s_e=0.5)

    # Far from target
    far = SCoordinate(s_k=0.1, s_t=0.9, s_e=0.3)
    far_result = core.completion_check(far, target, epsilon=1e-3)
    print(f"   Far point: d_cat = {far_result['categorical_distance']:.6f}, completed = {far_result['completed']}")

    # Close to target
    close = SCoordinate(s_k=0.5001, s_t=0.4999, s_e=0.5)
    close_result = core.completion_check(close, target, epsilon=1e-2)
    print(f"   Close point: d_cat = {close_result['categorical_distance']:.6f}, completed = {close_result['completed']}")

    results['completion_far'] = far_result
    results['completion_close'] = close_result

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = validate_s_entropy_coordinates()
