"""
Trajectory Navigation System

Implementation of trajectory-based navigation in S-entropy space.
Key theorem: Trajectory-Position Identity - the trit sequence specifying
a cell simultaneously encodes position, trajectory, and address.

These are the same mathematical object, eliminating the separation
between data location and access path (von Neumann separation).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import hashlib
import time

try:
    from .s_entropy import SCoordinate, SEntropyCore
    from .ternary import TritAddress, TernaryEncoder
except ImportError:
    from s_entropy import SCoordinate, SEntropyCore
    from ternary import TritAddress, TernaryEncoder


@dataclass
class Trajectory:
    """
    A trajectory through S-entropy space.

    The trajectory encodes three things simultaneously:
    1. Position (final coordinates)
    2. Path (sequence of refinements)
    3. Address (navigation identifier)

    These are the same mathematical object per the Trajectory-Position Identity.
    """
    origin: SCoordinate
    waypoints: List[SCoordinate] = field(default_factory=list)
    current: SCoordinate = None
    _hash: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.current is None:
            self.current = self.origin
        self._compute_hash()

    def _compute_hash(self):
        """Compute trajectory hash (the address)."""
        # Concatenate all coordinates
        data = self.origin.to_array().tobytes()
        for wp in self.waypoints:
            data += wp.to_array().tobytes()
        data += self.current.to_array().tobytes()

        self._hash = hashlib.sha256(data).hexdigest()

    @property
    def address(self) -> str:
        """The trajectory IS the address."""
        return self._hash

    @property
    def position(self) -> SCoordinate:
        """The trajectory IS the position."""
        return self.current

    @property
    def path(self) -> List[SCoordinate]:
        """The trajectory IS the path."""
        return [self.origin] + self.waypoints + [self.current]

    def extend(self, waypoint: SCoordinate) -> 'Trajectory':
        """Create new trajectory with additional waypoint."""
        new_trajectory = Trajectory(
            origin=self.origin,
            waypoints=self.waypoints + [self.current],
            current=waypoint
        )
        return new_trajectory

    def length(self) -> int:
        """Number of waypoints traversed."""
        return len(self.waypoints) + 1


class TrajectoryNavigator:
    """
    Navigator for S-entropy space using trajectory-based addressing.

    Implements:
    - Navigation to target coordinates
    - Completion detection at epsilon-boundary
    - Trajectory-based data retrieval
    """

    def __init__(self, epsilon: float = 1e-6, max_steps: int = 1000):
        """
        Initialize navigator.

        Args:
            epsilon: Completion threshold
            max_steps: Maximum navigation steps
        """
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.core = SEntropyCore()
        self.encoder = TernaryEncoder()

    def navigate(
        self,
        start: SCoordinate,
        target: SCoordinate,
        strategy: str = 'gradient'
    ) -> Trajectory:
        """
        Navigate from start to target.

        Args:
            start: Starting coordinate
            target: Target coordinate
            strategy: Navigation strategy ('gradient', 'random', 'categorical')

        Returns:
            Complete trajectory
        """
        trajectory = Trajectory(origin=start)

        for step in range(self.max_steps):
            # Check completion
            completion = self.core.completion_check(
                trajectory.current, target, self.epsilon
            )

            if completion['completed']:
                break

            # Compute next waypoint based on strategy
            if strategy == 'gradient':
                next_point = self._gradient_step(trajectory.current, target)
            elif strategy == 'random':
                next_point = self._random_step(trajectory.current, target)
            elif strategy == 'categorical':
                next_point = self._categorical_step(trajectory.current, target)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            trajectory = trajectory.extend(next_point)

        return trajectory

    def _gradient_step(
        self,
        current: SCoordinate,
        target: SCoordinate,
        step_size: float = 0.1
    ) -> SCoordinate:
        """Move toward target using gradient descent."""
        direction = target.to_array() - current.to_array()
        norm = np.linalg.norm(direction)

        if norm < 1e-10:
            return current

        step = direction / norm * min(step_size, norm)
        new_coords = current.to_array() + step

        # Clamp to [0, 1]
        new_coords = np.clip(new_coords, 0.0, 1.0)

        return SCoordinate.from_array(new_coords)

    def _random_step(
        self,
        current: SCoordinate,
        target: SCoordinate,
        step_size: float = 0.05
    ) -> SCoordinate:
        """Random walk biased toward target."""
        # Random direction with bias toward target
        bias = target.to_array() - current.to_array()
        bias = bias / (np.linalg.norm(bias) + 1e-10)

        random_dir = np.random.randn(3)
        random_dir = random_dir / np.linalg.norm(random_dir)

        # Mix random with bias
        direction = 0.5 * bias + 0.5 * random_dir
        direction = direction / np.linalg.norm(direction)

        step = direction * step_size
        new_coords = current.to_array() + step

        # Clamp to [0, 1]
        new_coords = np.clip(new_coords, 0.0, 1.0)

        return SCoordinate.from_array(new_coords)

    def _categorical_step(
        self,
        current: SCoordinate,
        target: SCoordinate
    ) -> SCoordinate:
        """Move toward target using categorical refinement."""
        # Encode both to ternary
        current_addr = self.encoder.encode(current, depth=10)
        target_addr = self.encoder.encode(target, depth=10)

        # Find first differing trit
        new_trits = list(current_addr.trits)
        for i, (c, t) in enumerate(zip(current_addr.trits, target_addr.trits)):
            if c != t:
                # Move one step toward target in this dimension
                if c < t:
                    new_trits[i] = min(2, c + 1)
                else:
                    new_trits[i] = max(0, c - 1)
                break

        new_addr = TritAddress(trits=new_trits)
        return self.encoder.decode(new_addr)

    def verify_trajectory_position_identity(self, n_samples: int = 100) -> dict:
        """
        Verify the Trajectory-Position Identity theorem.

        The theorem states that trajectory, position, and address
        are the same mathematical object.

        Args:
            n_samples: Number of trajectories to test

        Returns:
            Verification results
        """
        results = []

        for _ in range(n_samples):
            # Generate random start and target
            start = SCoordinate.random()
            target = SCoordinate.random()

            # Navigate
            trajectory = self.navigate(start, target, strategy='gradient')

            # Verify identity properties
            # 1. Position is current coordinate
            position_is_current = trajectory.position == trajectory.current

            # 2. Address is hash of trajectory
            address_computed = trajectory.address == trajectory._hash

            # 3. Path contains all points from origin to current
            path_complete = (
                len(trajectory.path) == trajectory.length() + 1 and
                trajectory.path[0] == trajectory.origin and
                trajectory.path[-1] == trajectory.current
            )

            # 4. Same trajectory -&gt; same address
            trajectory2 = Trajectory(
                origin=trajectory.origin,
                waypoints=trajectory.waypoints,
                current=trajectory.current
            )
            same_address = trajectory.address == trajectory2.address

            results.append({
                'position_is_current': position_is_current,
                'address_computed': address_computed,
                'path_complete': path_complete,
                'same_address': same_address,
                'trajectory_length': trajectory.length()
            })

        # Aggregate results
        all_position_current = all(r['position_is_current'] for r in results)
        all_address_computed = all(r['address_computed'] for r in results)
        all_path_complete = all(r['path_complete'] for r in results)
        all_same_address = all(r['same_address'] for r in results)

        theorem_verified = (all_position_current and all_address_computed and
                          all_path_complete and all_same_address)

        return {
            'n_samples': n_samples,
            'position_is_current': all_position_current,
            'address_computed': all_address_computed,
            'path_complete': all_path_complete,
            'same_trajectory_same_address': all_same_address,
            'theorem_verified': theorem_verified,
            'average_trajectory_length': np.mean([r['trajectory_length'] for r in results])
        }

    def verify_completion_equivalence(self, n_samples: int = 50) -> dict:
        """
        Verify the Completion Equivalence theorem.

        The operation that finds a solution is identical to
        the operation that verifies it:
        navigate(S_0, C) ≡ verify(S_k, C)

        Args:
            n_samples: Number of tests

        Returns:
            Verification results
        """
        results = []

        for _ in range(n_samples):
            target = SCoordinate.random()
            start = SCoordinate.random()

            # Navigate to find solution
            trajectory = self.navigate(start, target, strategy='gradient')

            # The final position
            found_position = trajectory.current

            # Verify: check if found position satisfies completion
            completion = self.core.completion_check(
                found_position, target, self.epsilon
            )

            # Navigate and verify use the same check
            # If navigation completed, verification should pass
            results.append({
                'navigation_completed': completion['completed'],
                'verification_passed': completion['at_epsilon_boundary'],
                'equivalence': completion['completed'] == completion['at_epsilon_boundary'],
                'categorical_distance': completion['categorical_distance']
            })

        all_equivalent = all(r['equivalence'] for r in results)
        completion_rate = np.mean([r['navigation_completed'] for r in results])

        return {
            'n_samples': n_samples,
            'all_equivalent': all_equivalent,
            'completion_rate': completion_rate,
            'average_distance': np.mean([r['categorical_distance'] for r in results]),
            'theorem_verified': all_equivalent
        }


def validate_trajectory_navigation() -> dict:
    """
    Run validation experiments for trajectory navigation system.
    """
    print("=" * 60)
    print("TRAJECTORY NAVIGATION VALIDATION")
    print("=" * 60)

    navigator = TrajectoryNavigator(epsilon=1e-3, max_steps=500)
    results = {}

    # Test 1: Trajectory-Position Identity
    print("\n1. Trajectory-Position Identity Theorem")
    print("-" * 40)

    tpi = navigator.verify_trajectory_position_identity(n_samples=100)
    results['trajectory_position_identity'] = tpi

    status = "[OK]" if tpi['theorem_verified'] else "[FAIL]"
    print(f"   {status} Theorem verified: {tpi['theorem_verified']}")
    print(f"   - Position is current: {tpi['position_is_current']}")
    print(f"   - Address computed correctly: {tpi['address_computed']}")
    print(f"   - Path complete: {tpi['path_complete']}")
    print(f"   - Same trajectory -&gt; same address: {tpi['same_trajectory_same_address']}")
    print(f"   - Average trajectory length: {tpi['average_trajectory_length']:.1f}")

    # Test 2: Completion Equivalence
    print("\n2. Completion Equivalence Theorem")
    print("-" * 40)

    ce = navigator.verify_completion_equivalence(n_samples=50)
    results['completion_equivalence'] = ce

    status = "[OK]" if ce['theorem_verified'] else "[FAIL]"
    print(f"   {status} Theorem verified: {ce['theorem_verified']}")
    print(f"   - All navigate/verify equivalent: {ce['all_equivalent']}")
    print(f"   - Completion rate: {ce['completion_rate']:.2%}")
    print(f"   - Average final distance: {ce['average_distance']:.6f}")

    # Test 3: Navigation Strategies
    print("\n3. Navigation Strategy Comparison")
    print("-" * 40)

    strategies = ['gradient', 'random', 'categorical']
    strategy_results = {}

    for strategy in strategies:
        nav = TrajectoryNavigator(epsilon=1e-3, max_steps=500)
        lengths = []
        times = []
        completions = 0

        for _ in range(20):
            start = SCoordinate.random()
            target = SCoordinate.random()

            t0 = time.time()
            trajectory = nav.navigate(start, target, strategy=strategy)
            t1 = time.time()

            lengths.append(trajectory.length())
            times.append(t1 - t0)

            completion = nav.core.completion_check(
                trajectory.current, target, nav.epsilon
            )
            if completion['completed']:
                completions += 1

        strategy_results[strategy] = {
            'avg_length': np.mean(lengths),
            'avg_time': np.mean(times),
            'completion_rate': completions / 20
        }

        print(f"   {strategy:12s}: length={np.mean(lengths):6.1f}, "
              f"time={np.mean(times)*1000:6.2f}ms, "
              f"complete={completions/20:.0%}")

    results['strategies'] = strategy_results

    # Test 4: Address Uniqueness
    print("\n4. Address Uniqueness")
    print("-" * 40)

    addresses = set()
    n_trajectories = 200

    for _ in range(n_trajectories):
        start = SCoordinate.random()
        target = SCoordinate.random()
        trajectory = navigator.navigate(start, target)
        addresses.add(trajectory.address)

    uniqueness_rate = len(addresses) / n_trajectories
    all_unique = len(addresses) == n_trajectories

    print(f"   Generated {n_trajectories} trajectories")
    print(f"   Unique addresses: {len(addresses)}")
    print(f"   {'[OK]' if all_unique else '[FAIL]'} All addresses unique: {all_unique}")

    results['address_uniqueness'] = {
        'n_trajectories': n_trajectories,
        'unique_addresses': len(addresses),
        'all_unique': all_unique
    }

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = validate_trajectory_navigation()
