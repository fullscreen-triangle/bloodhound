#!/usr/bin/env python3
"""
Genome Validation Experiment

Comprehensive validation experiment for the distributed genomics network
demonstrating S-entropy coordinate transformation, precision-by-difference
synchronization, and proof-based search algorithms.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.sequence import (
    SequenceValidationSuite,
    transform_sequence_to_coordinates,
    compare_with_traditional_methods
)
from src.genome import (
    GenomeValidationSuite,
    transform_genome_to_coordinates,
    compare_genome_assembly_methods
)
from src.network import (
    create_precision_network,
    analyze_distributed
)
from src.search import (
    search_genome_space,
    GraffitiGenomeSearch,
    ProofValidatedCompression,
    BatchAmbiguousCompression
)


class GenomeSystemValidator:
    """
    Comprehensive validator for the entire genome system.

    Validates all components: S-entropy transformation, distributed network,
    and proof-based search algorithms.
    """

    def __init__(self):
        self.results = {}
        self.figures = []

    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation of the genome system."""
        print("=" * 80)
        print("GENOME SYSTEM VALIDATION EXPERIMENT")
        print("=" * 80)
        print()

        # 1. Sequence Validation
        print("1. Running Sequence S-Entropy Validation...")
        sequence_results = self._validate_sequence_transformation()
        self.results['sequence'] = sequence_results
        print("   ✓ Sequence validation completed")
        print()

        # 2. Genome Validation
        print("2. Running Genome S-Entropy Validation...")
        genome_results = self._validate_genome_transformation()
        self.results['genome'] = genome_results
        print("   ✓ Genome validation completed")
        print()

        # 3. Network Validation
        print("3. Running Distributed Network Validation...")
        network_results = self._validate_distributed_network()
        self.results['network'] = network_results
        print("   ✓ Network validation completed")
        print()

        # 4. Search Validation
        print("4. Running Graffiti Search Validation...")
        search_results = self._validate_search_algorithms()
        self.results['search'] = search_results
        print("   ✓ Search validation completed")
        print()

        # 5. Compression Validation
        print("5. Running Compression Algorithm Validation...")
        compression_results = self._validate_compression_algorithms()
        self.results['compression'] = compression_results
        print("   ✓ Compression validation completed")
        print()

        # 6. Integration Validation
        print("6. Running System Integration Validation...")
        integration_results = self._validate_system_integration()
        self.results['integration'] = integration_results
        print("   ✓ Integration validation completed")
        print()

        # Generate comprehensive report
        self._generate_validation_report()

        # Create visualizations
        self._create_validation_visualizations()

        return self.results

    def _validate_sequence_transformation(self) -> Dict[str, Any]:
        """Validate S-entropy sequence transformation."""
        validator = SequenceValidationSuite()

        # Run validation with different sequence counts
        sequence_counts = [50, 100, 200]
        validation_results = []

        for count in sequence_counts:
            print(f"   Testing with {count} sequences...")
            results = validator.run_full_validation(count)
            validation_results.append({
                'sequence_count': count,
                'results': results
            })

        # Compare with traditional methods
        test_sequences = validator.generate_test_sequences(100)
        comparison_results = compare_with_traditional_methods(test_sequences)

        return {
            'validation_results': validation_results,
            'comparison_results': comparison_results,
            'test_sequences_count': len(test_sequences)
        }

    def _validate_genome_transformation(self) -> Dict[str, Any]:
        """Validate S-entropy genome transformation."""
        validator = GenomeValidationSuite()

        # Run validation with different genome counts
        genome_counts = [5, 10, 15]
        validation_results = []

        for count in genome_counts:
            print(f"   Testing with {count} genomes...")
            results = validator.run_full_genome_validation(count)
            validation_results.append({
                'genome_count': count,
                'results': results
            })

        # Test specific genome transformation
        test_genomes = validator.generate_test_genomes(5)
        transformation_results = []

        for i, genome in enumerate(test_genomes):
            genome_id = f"validation_genome_{i}"
            coords = transform_genome_to_coordinates(genome, genome_id)
            transformation_results.append({
                'genome_id': genome_id,
                'coordinate_dimensions': len(coords.global_coordinates),
                'total_s_entropy': coords.total_s_entropy,
                'chromosome_count': len(coords.chromosomes)
            })

        return {
            'validation_results': validation_results,
            'transformation_results': transformation_results,
            'test_genomes_count': len(test_genomes)
        }

    def _validate_distributed_network(self) -> Dict[str, Any]:
        """Validate distributed genomics network."""
        # Create network
        network = create_precision_network()

        # Join multiple nodes
        node_ids = []
        for i in range(5):
            node_id = network.join_network(node_capacity=0.8)
            node_ids.append(node_id)
            print(f"   Node {i+1} joined: {node_id}")

        # Test network operations
        test_results = []

        # Test 1: Sequence analysis
        test_sequence = "ATCGATCGATCGATCG" * 10
        query_data = {'sequence': test_sequence}

        start_time = time.time()
        result = analyze_distributed(network, query_data)
        analysis_time = time.time() - start_time

        test_results.append({
            'test_type': 'sequence_analysis',
            'analysis_time': analysis_time,
            'result_keys': list(result.keys()),
            'information_content': result.get('total_information_content', 0)
        })\n        \n        # Test 2: Genome analysis\n        test_genome = {\n            'chr1': 'ATCGATCG' * 1000,\n            'chr2': 'GCTAGCTA' * 800\n        }\n        query_data = {'genome': test_genome}\n        \n        start_time = time.time()\n        result = analyze_distributed(network, query_data)\n        analysis_time = time.time() - start_time\n        \n        test_results.append({\n            'test_type': 'genome_analysis',\n            'analysis_time': analysis_time,\n            'result_keys': list(result.keys()),\n            'information_content': result.get('total_information_content', 0)\n        })\n        \n        # Get network status\n        network_status = network.get_network_status()\n        \n        # Clean up nodes\n        for node_id in node_ids:\n            network.leave_network(node_id)\n        \n        return {\n            'test_results': test_results,\n            'network_status': network_status,\n            'nodes_tested': len(node_ids)\n        }\n    \n    def _validate_search_algorithms(self) -> Dict[str, Any]:\n        \"\"\"Validate Graffiti-based search algorithms.\"\"\"\n        # Create search engine\n        searcher = GraffitiGenomeSearch()\n        \n        # Test different query types\n        test_queries = [\n            ('ATCGATCGATCGATCG', 'sequence'),\n            ('human hemoglobin gene', 'function'),\n            ('diabetes associated variants', 'phenotype')\n        ]\n        \n        search_results = []\n        \n        for query, query_type in test_queries:\n            print(f\"   Testing {query_type} search: {query[:30]}...\")\n            \n            start_time = time.time()\n            results = searcher.search_genome_space(\n                query=query,\n                query_type=query_type,\n                compression_level='hybrid',\n                max_results=5\n            )\n            search_time = time.time() - start_time\n            \n            search_results.append({\n                'query': query,\n                'query_type': query_type,\n                'search_time': search_time,\n                'result_count': len(results),\n                'avg_relevance': np.mean([r.relevance_score for r in results]) if results else 0,\n                'avg_confidence': np.mean([r.confidence_level for r in results]) if results else 0\n            })\n        \n        # Test search history\n        search_history = searcher.get_search_history()\n        \n        return {\n            'search_results': search_results,\n            'search_history_count': len(search_history),\n            'total_queries_tested': len(test_queries)\n        }\n    \n    def _validate_compression_algorithms(self) -> Dict[str, Any]:\n        \"\"\"Validate compression algorithms.\"\"\"\n        # Test proof-validated compression\n        proof_compressor = ProofValidatedCompression()\n        \n        test_data = {\n            'sequence': 'ATCGATCGATCGATCG' * 100,\n            'metadata': {'source': 'test', 'length': 1600}\n        }\n        \n        # Proof-validated compression test\n        start_time = time.time()\n        compressed_data, proof_validation = proof_compressor.compress_with_proof(test_data)\n        compression_time = time.time() - start_time\n        \n        # Decompression test\n        start_time = time.time()\n        decompressed_data = proof_compressor.decompress_with_verification(compressed_data, proof_validation)\n        decompression_time = time.time() - start_time\n        \n        proof_results = {\n            'compression_time': compression_time,\n            'decompression_time': decompression_time,\n            'compression_ratio': proof_validation['compression_ratio'],\n            'validation_score': proof_validation['validation_score'],\n            'proof_verified': proof_validation['proof_verified'],\n            'data_integrity': decompressed_data == test_data\n        }\n        \n        # Test batch-ambiguous compression\n        batch_compressor = BatchAmbiguousCompression()\n        \n        batch_data = [\n            {'sequence': 'ATCG' * 50, 'type': 'gene'},\n            {'sequence': 'GCTA' * 50, 'type': 'gene'},\n            {'sequence': 'TTAA' * 50, 'type': 'intergenic'}\n        ]\n        \n        start_time = time.time()\n        compressed_batch, batch_metadata = batch_compressor.compress_batch_ambiguous(batch_data)\n        batch_compression_time = time.time() - start_time\n        \n        start_time = time.time()\n        decompressed_batch = batch_compressor.decompress_batch_ambiguous(compressed_batch, batch_metadata)\n        batch_decompression_time = time.time() - start_time\n        \n        batch_results = {\n            'compression_time': batch_compression_time,\n            'decompression_time': batch_decompression_time,\n            'compression_ratio': batch_metadata['compression_ratio'],\n            'ambiguity_level': batch_metadata['ambiguity_level'],\n            'ambiguity_groups': batch_metadata['ambiguity_groups'],\n            'data_integrity': len(decompressed_batch) == len(batch_data)\n        }\n        \n        return {\n            'proof_validated': proof_results,\n            'batch_ambiguous': batch_results\n        }\n    \n    def _validate_system_integration(self) -> Dict[str, Any]:\n        \"\"\"Validate integration of all system components.\"\"\"\n        # Create integrated system\n        network = create_precision_network()\n        searcher = GraffitiGenomeSearch(network=network)\n        \n        # Join nodes to network\n        node_ids = []\n        for i in range(3):\n            node_id = network.join_network()\n            node_ids.append(node_id)\n        \n        # Test integrated workflow\n        test_sequence = \"ATCGATCGATCGATCG\" * 20\n        \n        # Step 1: Transform sequence to coordinates\n        start_time = time.time()\n        seq_coords = transform_sequence_to_coordinates(test_sequence)\n        transform_time = time.time() - start_time\n        \n        # Step 2: Search using distributed network\n        start_time = time.time()\n        search_results = searcher.search_genome_space(\n            query=test_sequence,\n            query_type='sequence',\n            compression_level='hybrid',\n            max_results=3\n        )\n        search_time = time.time() - start_time\n        \n        # Step 3: Analyze results on network\n        if search_results:\n            query_data = {\n                'coordinates': search_results[0].genomic_coordinates.tolist()\n            }\n            \n            start_time = time.time()\n            analysis_result = analyze_distributed(network, query_data)\n            analysis_time = time.time() - start_time\n        else:\n            analysis_result = {}\n            analysis_time = 0\n        \n        # Clean up\n        for node_id in node_ids:\n            network.leave_network(node_id)\n        \n        return {\n            'transform_time': transform_time,\n            'search_time': search_time,\n            'analysis_time': analysis_time,\n            'total_workflow_time': transform_time + search_time + analysis_time,\n            'search_results_count': len(search_results),\n            'coordinate_dimensions': len(seq_coords.coordinates),\n            's_entropy': seq_coords.s_entropy,\n            'workflow_success': len(search_results) > 0 and len(analysis_result) > 0\n        }\n    \n    def _generate_validation_report(self):\n        \"\"\"Generate comprehensive validation report.\"\"\"\n        print(\"\\n\" + \"=\" * 80)\n        print(\"VALIDATION REPORT\")\n        print(\"=\" * 80)\n        \n        # Sequence validation summary\n        if 'sequence' in self.results:\n            seq_results = self.results['sequence']\n            print(\"\\nSEQUENCE TRANSFORMATION:\")\n            print(\"-\" * 30)\n            \n            if seq_results['validation_results']:\n                latest_validation = seq_results['validation_results'][-1]['results']\n                print(f\"Coordinate consistency: {latest_validation.accuracy_metrics.get('coordinate_consistency', 0):.4f}\")\n                print(f\"Average transformation time: {latest_validation.performance_metrics.get('avg_transformation_time', 0):.4f}s\")\n                print(f\"Nucleotides per second: {latest_validation.performance_metrics.get('nucleotides_per_second', 0):.0f}\")\n        \n        # Genome validation summary\n        if 'genome' in self.results:\n            genome_results = self.results['genome']\n            print(\"\\nGENOME TRANSFORMATION:\")\n            print(\"-\" * 25)\n            \n            if genome_results['validation_results']:\n                latest_validation = genome_results['validation_results'][-1]['results']\n                print(f\"Coordinate consistency: {latest_validation.accuracy_metrics.get('coordinate_consistency', 0):.4f}\")\n                print(f\"Average transformation time: {latest_validation.performance_metrics.get('avg_transformation_time', 0):.4f}s\")\n                \n            if genome_results['transformation_results']:\n                avg_s_entropy = np.mean([r['total_s_entropy'] for r in genome_results['transformation_results']])\n                print(f\"Average S-entropy: {avg_s_entropy:.4f}\")\n        \n        # Network validation summary\n        if 'network' in self.results:\n            network_results = self.results['network']\n            print(\"\\nDISTRIBUTED NETWORK:\")\n            print(\"-\" * 20)\n            \n            if network_results['test_results']:\n                avg_analysis_time = np.mean([r['analysis_time'] for r in network_results['test_results']])\n                print(f\"Average analysis time: {avg_analysis_time:.4f}s\")\n                print(f\"Nodes tested: {network_results['nodes_tested']}\")\n                print(f\"Network capacity: {network_results['network_status']['total_processing_capacity']:.2f}\")\n        \n        # Search validation summary\n        if 'search' in self.results:\n            search_results = self.results['search']\n            print(\"\\nGRAFFITI SEARCH:\")\n            print(\"-\" * 16)\n            \n            if search_results['search_results']:\n                avg_search_time = np.mean([r['search_time'] for r in search_results['search_results']])\n                avg_relevance = np.mean([r['avg_relevance'] for r in search_results['search_results']])\n                print(f\"Average search time: {avg_search_time:.4f}s\")\n                print(f\"Average relevance score: {avg_relevance:.4f}\")\n                print(f\"Queries tested: {search_results['total_queries_tested']}\")\n        \n        # Compression validation summary\n        if 'compression' in self.results:\n            comp_results = self.results['compression']\n            print(\"\\nCOMPRESSION ALGORITHMS:\")\n            print(\"-\" * 23)\n            \n            proof_results = comp_results['proof_validated']\n            batch_results = comp_results['batch_ambiguous']\n            \n            print(f\"Proof-validated compression ratio: {proof_results['compression_ratio']:.4f}\")\n            print(f\"Proof validation score: {proof_results['validation_score']:.4f}\")\n            print(f\"Batch compression ratio: {batch_results['compression_ratio']:.4f}\")\n            print(f\"Batch ambiguity level: {batch_results['ambiguity_level']:.4f}\")\n        \n        # Integration summary\n        if 'integration' in self.results:\n            int_results = self.results['integration']\n            print(\"\\nSYSTEM INTEGRATION:\")\n            print(\"-\" * 19)\n            \n            print(f\"Total workflow time: {int_results['total_workflow_time']:.4f}s\")\n            print(f\"Workflow success: {int_results['workflow_success']}\")\n            print(f\"S-entropy value: {int_results['s_entropy']:.4f}\")\n        \n        print(\"\\n\" + \"=\" * 80)\n    \n    def _create_validation_visualizations(self):\n        \"\"\"Create validation visualizations.\"\"\"\n        # Set up plotting style\n        plt.style.use('seaborn-v0_8')\n        sns.set_palette(\"husl\")\n        \n        # Create figure with subplots\n        fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n        fig.suptitle('Genome System Validation Results', fontsize=16, fontweight='bold')\n        \n        # Plot 1: Sequence transformation performance\n        if 'sequence' in self.results:\n            ax = axes[0, 0]\n            seq_results = self.results['sequence']['validation_results']\n            \n            sequence_counts = [r['sequence_count'] for r in seq_results]\n            transform_times = [r['results'].performance_metrics.get('avg_transformation_time', 0) for r in seq_results]\n            \n            ax.plot(sequence_counts, transform_times, 'o-', linewidth=2, markersize=8)\n            ax.set_xlabel('Number of Sequences')\n            ax.set_ylabel('Avg Transform Time (s)')\n            ax.set_title('Sequence Transformation Scalability')\n            ax.grid(True, alpha=0.3)\n        \n        # Plot 2: Genome transformation performance\n        if 'genome' in self.results:\n            ax = axes[0, 1]\n            genome_results = self.results['genome']['validation_results']\n            \n            genome_counts = [r['genome_count'] for r in genome_results]\n            transform_times = [r['results'].performance_metrics.get('avg_transformation_time', 0) for r in genome_results]\n            \n            ax.plot(genome_counts, transform_times, 's-', linewidth=2, markersize=8)\n            ax.set_xlabel('Number of Genomes')\n            ax.set_ylabel('Avg Transform Time (s)')\n            ax.set_title('Genome Transformation Scalability')\n            ax.grid(True, alpha=0.3)\n        \n        # Plot 3: Network analysis performance\n        if 'network' in self.results:\n            ax = axes[0, 2]\n            network_results = self.results['network']['test_results']\n            \n            test_types = [r['test_type'] for r in network_results]\n            analysis_times = [r['analysis_time'] for r in network_results]\n            \n            bars = ax.bar(test_types, analysis_times, alpha=0.7)\n            ax.set_ylabel('Analysis Time (s)')\n            ax.set_title('Network Analysis Performance')\n            ax.tick_params(axis='x', rotation=45)\n            \n            # Add value labels on bars\n            for bar, time_val in zip(bars, analysis_times):\n                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,\n                       f'{time_val:.3f}', ha='center', va='bottom')\n        \n        # Plot 4: Search performance\n        if 'search' in self.results:\n            ax = axes[1, 0]\n            search_results = self.results['search']['search_results']\n            \n            query_types = [r['query_type'] for r in search_results]\n            relevance_scores = [r['avg_relevance'] for r in search_results]\n            \n            bars = ax.bar(query_types, relevance_scores, alpha=0.7, color='orange')\n            ax.set_ylabel('Average Relevance Score')\n            ax.set_title('Search Relevance by Query Type')\n            ax.set_ylim(0, 1)\n            \n            # Add value labels\n            for bar, score in zip(bars, relevance_scores):\n                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n                       f'{score:.3f}', ha='center', va='bottom')\n        \n        # Plot 5: Compression performance\n        if 'compression' in self.results:\n            ax = axes[1, 1]\n            comp_results = self.results['compression']\n            \n            methods = ['Proof Validated', 'Batch Ambiguous']\n            ratios = [\n                comp_results['proof_validated']['compression_ratio'],\n                comp_results['batch_ambiguous']['compression_ratio']\n            ]\n            \n            bars = ax.bar(methods, ratios, alpha=0.7, color='green')\n            ax.set_ylabel('Compression Ratio')\n            ax.set_title('Compression Algorithm Performance')\n            \n            # Add value labels\n            for bar, ratio in zip(bars, ratios):\n                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n                       f'{ratio:.3f}', ha='center', va='bottom')\n        \n        # Plot 6: System integration workflow\n        if 'integration' in self.results:\n            ax = axes[1, 2]\n            int_results = self.results['integration']\n            \n            workflow_steps = ['Transform', 'Search', 'Analysis']\n            step_times = [\n                int_results['transform_time'],\n                int_results['search_time'],\n                int_results['analysis_time']\n            ]\n            \n            bars = ax.bar(workflow_steps, step_times, alpha=0.7, color='purple')\n            ax.set_ylabel('Time (s)')\n            ax.set_title('Integrated Workflow Performance')\n            \n            # Add value labels\n            for bar, time_val in zip(bars, step_times):\n                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,\n                       f'{time_val:.3f}', ha='center', va='bottom')\n        \n        plt.tight_layout()\n        \n        # Save figure\n        plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')\n        print(f\"\\nValidation visualizations saved to: validation_results.png\")\n        \n        self.figures.append(fig)\n        \n        return fig\n\n\ndef main():\n    \"\"\"Main validation experiment function.\"\"\"\n    print(\"Starting Genome System Validation Experiment...\")\n    print(f\"Python version: {sys.version}\")\n    print(f\"Working directory: {os.getcwd()}\")\n    print()\n    \n    # Create validator\n    validator = GenomeSystemValidator()\n    \n    # Run validation\n    start_time = time.time()\n    results = validator.run_full_validation()\n    total_time = time.time() - start_time\n    \n    print(f\"\\nTotal validation time: {total_time:.2f} seconds\")\n    print(\"\\nValidation experiment completed successfully!\")\n    \n    # Show plots if in interactive mode\n    try:\n        plt.show()\n    except:\n        print(\"Note: Plots saved but not displayed (non-interactive environment)\")\n    \n    return results\n\n\nif __name__ == \"__main__\":\n    results = main()
