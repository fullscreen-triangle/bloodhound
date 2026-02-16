#!/usr/bin/env python3
"""
Bloodhound VM Validation Suite

Complete validation of the distributed virtual machine architecture
based on categorical navigation in bounded phase space.

This script runs all validation experiments for the theoretical
framework described in the paper.

Usage:
    python run_validation.py [--quick] [--module MODULE]

Options:
    --quick     Run quick validation (fewer samples)
    --module    Run only specific module validation
                (s_entropy, ternary, trajectory, memory, demon, distributed, enhancement)
"""

import sys
import os
import time
import argparse
from typing import Dict, Any
import json

# Add the validation directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_s_entropy_validation() -> Dict[str, Any]:
    """Validate S-entropy coordinate system."""
    from s_entropy import validate_s_entropy_coordinates
    return validate_s_entropy_coordinates()


def run_ternary_validation() -> Dict[str, Any]:
    """Validate ternary representation system."""
    from ternary import validate_ternary_representation
    return validate_ternary_representation()


def run_trajectory_validation() -> Dict[str, Any]:
    """Validate trajectory navigation system."""
    from trajectory import validate_trajectory_navigation
    return validate_trajectory_navigation()


def run_memory_validation() -> Dict[str, Any]:
    """Validate categorical memory system."""
    from categorical_memory import validate_categorical_memory
    return validate_categorical_memory()


def run_demon_validation() -> Dict[str, Any]:
    """Validate Maxwell demon controller."""
    from maxwell_demon import validate_maxwell_demon
    return validate_maxwell_demon()


def run_distributed_validation() -> Dict[str, Any]:
    """Validate distributed coordination system."""
    from distributed import validate_distributed_coordination
    return validate_distributed_coordination()


def run_enhancement_validation() -> Dict[str, Any]:
    """Validate enhancement mechanisms."""
    from enhancement import validate_enhancement_mechanisms
    return validate_enhancement_mechanisms()


MODULES = {
    's_entropy': run_s_entropy_validation,
    'ternary': run_ternary_validation,
    'trajectory': run_trajectory_validation,
    'memory': run_memory_validation,
    'demon': run_demon_validation,
    'distributed': run_distributed_validation,
    'enhancement': run_enhancement_validation,
}


def run_full_validation() -> Dict[str, Any]:
    """Run complete validation suite."""
    print("=" * 70)
    print("BLOODHOUND VM - COMPLETE VALIDATION SUITE")
    print("Categorical Navigation in Bounded Phase Space")
    print("=" * 70)
    print()

    start_time = time.time()
    results = {}

    module_order = [
        ('s_entropy', 'S-Entropy Coordinates'),
        ('ternary', 'Ternary Representation'),
        ('trajectory', 'Trajectory Navigation'),
        ('memory', 'Categorical Memory'),
        ('demon', 'Maxwell Demon Controller'),
        ('distributed', 'Distributed Coordination'),
        ('enhancement', 'Enhancement Mechanisms'),
    ]

    for module_key, module_name in module_order:
        print(f"\n{'='*70}")
        print(f"MODULE: {module_name}")
        print(f"{'='*70}")

        module_start = time.time()

        try:
            module_results = MODULES[module_key]()
            results[module_key] = {
                'status': 'success',
                'results': module_results,
                'duration': time.time() - module_start
            }
        except Exception as e:
            print(f"\nERROR in {module_name}: {e}")
            results[module_key] = {
                'status': 'error',
                'error': str(e),
                'duration': time.time() - module_start
            }

        print(f"\nModule completed in {results[module_key]['duration']:.2f}s")

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()

    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = len(results) - successful

    print(f"Total modules: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print()

    print("Module Status:")
    for module_key, module_name in module_order:
        status = results.get(module_key, {}).get('status', 'unknown')
        duration = results.get(module_key, {}).get('duration', 0)
        symbol = "[OK]" if status == 'success' else "[FAIL]"
        print(f"  {symbol} {module_name}: {status} ({duration:.2f}s)")

    # Key theorems verification
    print("\n" + "-" * 70)
    print("KEY THEOREMS VERIFIED:")
    print("-" * 70)

    theorems = [
        ('Triple Equivalence', 's_entropy', 'triple_equivalence_M=3,n=10', 'triple_equivalence_verified'),
        ('Distance Independence', 's_entropy', 'distance_independence', 'independent'),
        ('Trit-Cell Correspondence', 'ternary', 'trit_cell_depth_5', 'theorem_verified'),
        ('Continuous Emergence', 'ternary', 'continuous_emergence', 'theorem_verified'),
        ('Trajectory-Position Identity', 'trajectory', 'trajectory_position_identity', 'theorem_verified'),
        ('Completion Equivalence', 'trajectory', 'completion_equivalence', 'theorem_verified'),
        ('3^k Hierarchy', 'memory', 'hierarchy', 'theorem_verified'),
        ('Zero-Cost Sorting', 'demon', 'zero_cost_sorting', 'theorem_verified'),
        ('Observable Commutation', 'demon', 'commutation', 'commutation_verified'),
        ('Exponential Decay', 'distributed', 'exponential_decay', 'theorem_verified'),
        ('Central State Impossibility', 'distributed', 'central_state_impossibility', 'theorem_verified'),
    ]

    for theorem_name, module, result_key, verified_key in theorems:
        try:
            module_results = results.get(module, {}).get('results', {})
            theorem_result = module_results.get(result_key, {})
            verified = theorem_result.get(verified_key, False)
            symbol = "[OK]" if verified else "[FAIL]"
            print(f"  {symbol} {theorem_name}")
        except:
            print(f"  ? {theorem_name} (could not verify)")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return {
        'modules': results,
        'summary': {
            'total_modules': len(results),
            'successful': successful,
            'failed': failed,
            'total_time': total_time
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Bloodhound VM Validation Suite'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick validation with fewer samples'
    )
    parser.add_argument(
        '--module',
        type=str,
        choices=list(MODULES.keys()),
        help='Run only specific module validation'
    )
    parser.add_argument(
        '--json',
        type=str,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    if args.module:
        print(f"Running {args.module} validation only...")
        results = MODULES[args.module]()
    else:
        results = run_full_validation()

    if args.json:
        # Convert results to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif hasattr(obj, '__dict__'):
                return make_serializable(obj.__dict__)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        serializable = make_serializable(results)

        with open(args.json, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.json}")

    return results


if __name__ == "__main__":
    results = main()
