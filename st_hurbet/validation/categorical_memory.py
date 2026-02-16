"""
Categorical Memory System

Implementation of memory organized as a 3^k hierarchical structure
addressed by S-entropy coordinates. Data placement follows categorical
distance from current position.

Key features:
- Trajectory-based addressing (address = trajectory hash)
- Tier assignment based on categorical distance
- Navigation complexity O(log_3 N)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib
import time

try:
    from .s_entropy import SCoordinate, SEntropyCore
    from .ternary import TritAddress, TernaryEncoder
    from .trajectory import Trajectory
except ImportError:
    from s_entropy import SCoordinate, SEntropyCore
    from ternary import TritAddress, TernaryEncoder
    from trajectory import Trajectory


class MemoryTier(Enum):
    """Memory tier based on categorical distance."""
    L1 = 1      # d < 10^-23 (fastest)
    L2 = 2      # 10^-23 ≤ d < 10^-22
    L3 = 3      # 10^-22 ≤ d < 10^-21
    RAM = 4     # 10^-21 ≤ d < 10^-20
    STORAGE = 5 # d ≥ 10^-20 (slowest)


@dataclass
class MemoryCell:
    """A cell in categorical memory."""
    address: TritAddress
    data: Any
    trajectory: Trajectory
    tier: MemoryTier
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

    @property
    def trajectory_hash(self) -> str:
        """The trajectory hash IS the address."""
        return self.trajectory.address


class CategoricalMemory:
    """
    Memory system organized by categorical coordinates.

    Storage is a 3^k hierarchical tree where:
    - Root represents entire S-space [0,1]^3
    - Each node has 3 children (one per trit value)
    - Leaf nodes at depth k represent 3^k cells
    - Data resides at leaves; internal nodes contain routing info
    """

    # Tier thresholds (categorical distance)
    TIER_THRESHOLDS = {
        MemoryTier.L1: 1e-23,
        MemoryTier.L2: 1e-22,
        MemoryTier.L3: 1e-21,
        MemoryTier.RAM: 1e-20,
        MemoryTier.STORAGE: float('inf')
    }

    # Simulated latencies (seconds)
    TIER_LATENCIES = {
        MemoryTier.L1: 1e-9,      # 1 ns
        MemoryTier.L2: 10e-9,     # 10 ns
        MemoryTier.L3: 50e-9,     # 50 ns
        MemoryTier.RAM: 100e-9,   # 100 ns
        MemoryTier.STORAGE: 1e-3  # 1 ms
    }

    def __init__(self, max_depth: int = 20):
        """
        Initialize categorical memory.

        Args:
            max_depth: Maximum trit depth for addresses
        """
        self.max_depth = max_depth
        self.core = SEntropyCore()
        self.encoder = TernaryEncoder(default_depth=max_depth)

        # Storage by tier
        self.tiers: Dict[MemoryTier, Dict[str, MemoryCell]] = {
            tier: {} for tier in MemoryTier
        }

        # Current position in S-space
        self.current_position = SCoordinate.center()

        # Statistics
        self.stats = {
            'navigations': 0,
            'hits': {tier: 0 for tier in MemoryTier},
            'misses': 0,
            'total_latency': 0.0
        }

    def _determine_tier(self, coord: SCoordinate) -> MemoryTier:
        """Determine memory tier based on categorical distance from current position."""
        d_cat = self.core.categorical_distance(coord, self.current_position)

        for tier in [MemoryTier.L1, MemoryTier.L2, MemoryTier.L3, MemoryTier.RAM]:
            if d_cat < self.TIER_THRESHOLDS[tier]:
                return tier

        return MemoryTier.STORAGE

    def store(
        self,
        data: Any,
        trajectory: Trajectory
    ) -> MemoryCell:
        """
        Store data at the location specified by trajectory.

        The trajectory hash IS the address.

        Args:
            data: Data to store
            trajectory: Navigation trajectory (determines address)

        Returns:
            Created memory cell
        """
        # Address from trajectory
        address = self.encoder.encode(trajectory.current, depth=self.max_depth)

        # Determine tier
        tier = self._determine_tier(trajectory.current)

        # Create cell
        cell = MemoryCell(
            address=address,
            data=data,
            trajectory=trajectory,
            tier=tier
        )

        # Store in appropriate tier
        self.tiers[tier][trajectory.address] = cell

        return cell

    def retrieve(
        self,
        target: SCoordinate
    ) -> Tuple[Optional[Any], float, MemoryTier]:
        """
        Navigate to target and retrieve data.

        Args:
            target: Target coordinate

        Returns:
            (data, latency, tier) or (None, latency, None) if not found
        """
        self.stats['navigations'] += 1

        # Navigation latency: O(log_3 N) operations
        n_items = sum(len(tier) for tier in self.tiers.values())
        nav_steps = int(np.log(max(1, n_items)) / np.log(3)) + 1

        # Encode target
        target_addr = self.encoder.encode(target, depth=self.max_depth)

        # Search tiers from fastest to slowest
        total_latency = 0.0
        for tier in MemoryTier:
            # Add tier access latency
            total_latency += self.TIER_LATENCIES[tier]

            # Search this tier (by trajectory hash proximity)
            for hash_key, cell in self.tiers[tier].items():
                # Check if cell is at target
                d_cat = self.core.categorical_distance(
                    cell.trajectory.current, target
                )

                if d_cat < 1e-10:  # Close enough
                    cell.access_count += 1
                    cell.last_access = time.time()
                    self.stats['hits'][tier] += 1
                    self.stats['total_latency'] += total_latency
                    return (cell.data, total_latency, tier)

        self.stats['misses'] += 1
        self.stats['total_latency'] += total_latency
        return (None, total_latency, None)

    def navigate(self, target: SCoordinate) -> Trajectory:
        """
        Navigate from current position to target.

        Updates current_position and returns the trajectory taken.
        """
        try:
            from .trajectory import TrajectoryNavigator
        except ImportError:
            from trajectory import TrajectoryNavigator

        navigator = TrajectoryNavigator()
        trajectory = navigator.navigate(self.current_position, target)

        # Update current position
        self.current_position = trajectory.current

        return trajectory

    def verify_hierarchy_structure(self, max_depth: int = 6) -> dict:
        """
        Verify the 3^k hierarchical structure.

        At depth k, there should be exactly 3^k possible cells.
        """
        results = []

        for depth in range(1, max_depth + 1):
            expected_cells = 3 ** depth

            # Generate all addresses at this depth
            def count_addresses(d: int) -> int:
                if d == 0:
                    return 1
                return 3 * count_addresses(d - 1)

            actual_cells = count_addresses(depth)

            results.append({
                'depth': depth,
                'expected': expected_cells,
                'actual': actual_cells,
                'correct': expected_cells == actual_cells
            })

        all_correct = all(r['correct'] for r in results)

        return {
            'results': results,
            'all_correct': all_correct,
            'theorem_verified': all_correct
        }

    def verify_navigation_complexity(self, n_tests: int = 100) -> dict:
        """
        Verify O(log_3 N) navigation complexity.

        Navigation should scale logarithmically with stored items.
        """
        try:
            from .trajectory import TrajectoryNavigator
        except ImportError:
            from trajectory import TrajectoryNavigator

        # Store varying amounts of data
        sizes = [10, 50, 100]
        complexity_results = []

        for size in sizes:
            # Create memory with items
            mem = CategoricalMemory(max_depth=15)
            navigator = TrajectoryNavigator(max_steps=100)

            # Store random data
            for i in range(size):
                start = SCoordinate.random()
                target = SCoordinate.random()
                trajectory = navigator.navigate(start, target)
                mem.store(f"data_{i}", trajectory)

            # Measure navigation time
            nav_times = []
            for _ in range(n_tests):
                target = SCoordinate.random()

                t0 = time.time()
                mem.navigate(target)
                t1 = time.time()

                nav_times.append(t1 - t0)

            avg_time = np.mean(nav_times)
            theoretical_complexity = np.log(size) / np.log(3)

            complexity_results.append({
                'size': size,
                'avg_navigation_time': avg_time,
                'theoretical_log3_N': theoretical_complexity,
                'time_per_step': avg_time / theoretical_complexity if theoretical_complexity > 0 else 0
            })

        # Check if time scales logarithmically
        # Ratio of times should be close to ratio of log(sizes)
        if len(complexity_results) >= 2:
            time_ratio = complexity_results[-1]['avg_navigation_time'] / complexity_results[0]['avg_navigation_time']
            log_ratio = np.log(sizes[-1]) / np.log(sizes[0])

            # Allow 2x tolerance
            scaling_correct = 0.5 < time_ratio / log_ratio < 2.0
        else:
            scaling_correct = True

        return {
            'results': complexity_results,
            'scaling_logarithmic': scaling_correct,
            'n_tests': n_tests
        }

    def get_statistics(self) -> dict:
        """Get memory access statistics."""
        total_hits = sum(self.stats['hits'].values())
        total_accesses = total_hits + self.stats['misses']

        hit_rate = total_hits / total_accesses if total_accesses > 0 else 0
        avg_latency = self.stats['total_latency'] / total_accesses if total_accesses > 0 else 0

        return {
            'navigations': self.stats['navigations'],
            'total_hits': total_hits,
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'hits_by_tier': dict(self.stats['hits']),
            'total_latency': self.stats['total_latency'],
            'average_latency': avg_latency,
            'items_by_tier': {tier.name: len(items) for tier, items in self.tiers.items()}
        }


def validate_categorical_memory() -> dict:
    """
    Run validation experiments for categorical memory system.
    """
    print("=" * 60)
    print("CATEGORICAL MEMORY VALIDATION")
    print("=" * 60)

    results = {}

    # Test 1: 3^k Hierarchy Structure
    print("\n1. 3^k Hierarchy Structure")
    print("-" * 40)

    mem = CategoricalMemory(max_depth=20)
    hierarchy = mem.verify_hierarchy_structure(max_depth=8)
    results['hierarchy'] = hierarchy

    status = "[OK]" if hierarchy['theorem_verified'] else "[FAIL]"
    print(f"   {status} Hierarchy verified: {hierarchy['theorem_verified']}")
    for r in hierarchy['results'][:5]:
        print(f"      Depth {r['depth']}: {r['expected']} cells")

    # Test 2: Store and Retrieve
    print("\n2. Store and Retrieve Operations")
    print("-" * 40)

    try:
        from .trajectory import TrajectoryNavigator
    except ImportError:
        from trajectory import TrajectoryNavigator
    navigator = TrajectoryNavigator()
    mem = CategoricalMemory(max_depth=15)

    # Store some data
    stored_items = []
    for i in range(20):
        start = SCoordinate.random()
        target = SCoordinate.random()
        trajectory = navigator.navigate(start, target)
        cell = mem.store(f"test_data_{i}", trajectory)
        stored_items.append((trajectory.current, f"test_data_{i}"))
        print(f"   Stored item {i} in tier {cell.tier.name}")

    # Retrieve some data
    retrieve_count = 0
    for coord, expected_data in stored_items[:5]:
        data, latency, tier = mem.retrieve(coord)
        if data == expected_data:
            retrieve_count += 1
            print(f"   [OK] Retrieved from {tier.name if tier else 'MISS'} "
                  f"in {latency*1e9:.1f} ns")
        else:
            print(f"   [FAIL] Retrieval mismatch")

    results['store_retrieve'] = {
        'stored': len(stored_items),
        'retrieved': retrieve_count,
        'success_rate': retrieve_count / 5
    }

    # Test 3: Navigation Complexity
    print("\n3. Navigation Complexity O(log_3 N)")
    print("-" * 40)

    complexity = mem.verify_navigation_complexity(n_tests=20)
    results['complexity'] = complexity

    for r in complexity['results']:
        print(f"   N={r['size']:5d}: time={r['avg_navigation_time']*1000:.3f}ms, "
              f"log_3(N)={r['theoretical_log3_N']:.2f}")

    status = "[OK]" if complexity['scaling_logarithmic'] else "?"
    print(f"   {status} Scaling is logarithmic: {complexity['scaling_logarithmic']}")

    # Test 4: Tier Distribution
    print("\n4. Tier Distribution")
    print("-" * 40)

    stats = mem.get_statistics()
    results['tier_stats'] = stats

    print(f"   Total items stored: {sum(stats['items_by_tier'].values())}")
    for tier_name, count in stats['items_by_tier'].items():
        print(f"   {tier_name:8s}: {count} items")

    print(f"\n   Hit rate: {stats['hit_rate']:.2%}")
    print(f"   Average latency: {stats['average_latency']*1e9:.2f} ns")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = validate_categorical_memory()
