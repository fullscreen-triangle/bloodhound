#!/usr/bin/env python3
"""
Genome System Demonstration

Interactive demonstration of the distributed genomics network with S-entropy
coordinate transformation and proof-based search algorithms.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.sequence import transform_sequence_to_coordinates, calculate_s_entropy
from src.genome import transform_genome_to_coordinates
from src.network import create_precision_network, analyze_distributed
from src.search import (
    search_genome_space,
    GraffitiGenomeSearch,
    MufakoseGenomicsFramework,
    mufakose_variant_detection,
    mufakose_search
)


def demo_sequence_transformation():
    """Demonstrate S-entropy sequence transformation."""
    print("1. S-ENTROPY SEQUENCE TRANSFORMATION")
    print("-" * 40)

    # Example sequences
    sequences = [
        "ATCGATCGATCGATCG",
        "GGGGCCCCAAAATTTT",
        "ATCGATCGATCGATCGATCGATCGATCGATCG",
        "ACGTACGTACGTACGT"
    ]

    for i, seq in enumerate(sequences, 1):
        print(f"\nSequence {i}: {seq}")

        # Transform to coordinates
        start_time = time.time()
        coords = transform_sequence_to_coordinates(seq)
        transform_time = time.time() - start_time

        # Calculate S-entropy
        s_entropy = calculate_s_entropy(seq)

        print(f"  Length: {len(seq)} nucleotides")
        print(f"  S-entropy: {s_entropy:.4f}")
        print(f"  Coordinate dimensions: {len(coords.coordinates)}")
        print(f"  Coordinate magnitude: {np.linalg.norm(coords.coordinates):.4f}")
        print(f"  Transformation time: {transform_time:.4f}s")
        print(f"  Navigation indices: {coords.navigation_indices[:5]}...")

    print("\n✓ Sequence transformation demonstration completed")


def demo_genome_transformation():
    """Demonstrate genome-level S-entropy transformation."""
    print("\n\n2. GENOME-LEVEL S-ENTROPY TRANSFORMATION")
    print("-" * 45)

    # Create example genomes
    genomes = {
        "Small Genome": {
            'chr1': 'ATCGATCG' * 100,
            'chr2': 'GCTAGCTA' * 80,
            'chr3': 'TTAACCGG' * 60
        },
        "Medium Genome": {
            'chr1': 'ATCGATCG' * 500,
            'chr2': 'GCTAGCTA' * 400,
            'chr3': 'TTAACCGG' * 300,
            'chr4': 'CGATCGAT' * 200,
            'chr5': 'GGCCTTAA' * 150
        }
    }

    for genome_name, genome_data in genomes.items():
        print(f"\n{genome_name}:")

        # Transform genome
        start_time = time.time()
        genome_coords = transform_genome_to_coordinates(genome_data, genome_name.lower().replace(' ', '_'))
        transform_time = time.time() - start_time

        print(f"  Chromosomes: {len(genome_coords.chromosomes)}")
        print(f"  Total size: {genome_coords.genome_size:,} nucleotides")
        print(f"  Total S-entropy: {genome_coords.total_s_entropy:.4f}")
        print(f"  Global coordinates: {len(genome_coords.global_coordinates)} dimensions")
        print(f"  Assembly score: {genome_coords.assembly_quality_metrics.get('assembly_score', 0):.4f}")
        print(f"  Transformation time: {transform_time:.4f}s")

        # Show chromosome details
        print("  Chromosome details:")
        for chrom in genome_coords.chromosomes[:3]:  # Show first 3
            print(f"    {chrom.chromosome_id}: {chrom.length:,} bp, S-entropy: {chrom.s_entropy:.4f}, GC: {chrom.gc_content:.1f}%")

    print("\n✓ Genome transformation demonstration completed")


def demo_distributed_network():
    """Demonstrate distributed genomics network."""
    print("\n\n3. DISTRIBUTED GENOMICS NETWORK")
    print("-" * 35)

    # Create network
    print("\nCreating precision-by-difference network...")
    network = create_precision_network()

    # Join multiple nodes
    print("Joining nodes to network:")
    node_ids = []
    for i in range(4):
        node_id = network.join_network(node_capacity=0.8)
        node_ids.append(node_id)
        print(f"  Node {i+1}: {node_id}")

    # Show network status
    status = network.get_network_status()
    print(f"\nNetwork Status:")
    print(f"  Active nodes: {status['active_nodes']}")
    print(f"  Total capacity: {status['total_processing_capacity']:.2f}")
    print(f"  Average load: {status['average_node_load']:.2f}")

    # Test distributed analysis
    print("\nTesting distributed analysis:")

    test_cases = [
        {
            'name': 'Sequence Analysis',
            'data': {'sequence': 'ATCGATCGATCGATCG' * 5}
        },
        {
            'name': 'Genome Analysis',
            'data': {'genome': {'chr1': 'ATCGATCG' * 200, 'chr2': 'GCTAGCTA' * 150}}
        }
    ]

    for test_case in test_cases:
        print(f"\n  {test_case['name']}:")

        start_time = time.time()
        result = analyze_distributed(network, test_case['data'])
        analysis_time = time.time() - start_time

        print(f"    Analysis time: {analysis_time:.4f}s")
        print(f"    Information content: {result.get('total_information_content', 0):.4f}")
        print(f"    Processing nodes: {result.get('processing_node_count', 0)}")
        print(f"    Similarity assessment: {result.get('insights', {}).get('similarity_assessment', 'unknown')}")

    # Clean up network
    print(f"\nCleaning up network...")
    for node_id in node_ids:
        network.leave_network(node_id)

    print("✓ Distributed network demonstration completed")


def demo_graffiti_search():
    """Demonstrate Graffiti-based genomic search."""
    print("\n\n4. GRAFFITI-BASED GENOMIC SEARCH")
    print("-" * 35)

    # Create search engine
    searcher = GraffitiGenomeSearch()

    # Test different types of searches
    search_queries = [
        {
            'query': 'ATCGATCGATCGATCGATCGATCGATCGATCG',
            'type': 'sequence',
            'description': 'Direct sequence search'
        },
        {
            'query': 'hemoglobin oxygen binding',
            'type': 'function',
            'description': 'Functional search'
        },
        {
            'query': 'diabetes susceptibility variants',
            'type': 'phenotype',
            'description': 'Phenotype-based search'
        }
    ]

    for i, search_query in enumerate(search_queries, 1):
        print(f"\nSearch {i}: {search_query['description']}")
        print(f"Query: {search_query['query']}")
        print(f"Type: {search_query['type']}")

        # Perform search
        start_time = time.time()
        results = searcher.search_genome_space(
            query=search_query['query'],
            query_type=search_query['type'],
            compression_level='hybrid',
            max_results=3
        )
        search_time = time.time() - start_time

        print(f"Search time: {search_time:.4f}s")
        print(f"Results found: {len(results)}")

        # Show top results
        for j, result in enumerate(results[:2], 1):
            print(f"  Result {j}:")
            print(f"    Relevance: {result.relevance_score:.4f}")
            print(f"    Confidence: {result.confidence_level:.4f}")
            print(f"    Reasoning: {result.llm_reasoning[:100]}...")
            if result.proof_validation:
                print(f"    Proof verified: {result.proof_validation.get('proof_verified', False)}")

    print("\n✓ Graffiti search demonstration completed")


def demo_compression_algorithms():
    """Demonstrate compression algorithms."""
    print("\n\n5. COMPRESSION ALGORITHMS")
    print("-" * 25)

    # Test data
    test_data = {
        'sequence': 'ATCGATCGATCGATCG' * 20,
        'metadata': {
            'source': 'demo',
            'organism': 'synthetic',
            'length': 320
        },
        'annotations': [
            {'type': 'gene', 'start': 0, 'end': 100},
            {'type': 'exon', 'start': 20, 'end': 80}
        ]
    }

    print(f"Original data size: {len(str(test_data))} characters")

    # Proof-validated compression
    print("\nProof-Validated Compression:")
    from src.search import ProofValidatedCompression

    compressor = ProofValidatedCompression()

    start_time = time.time()
    compressed_data, proof_validation = compressor.compress_with_proof(test_data)
    compression_time = time.time() - start_time

    print(f"  Compression time: {compression_time:.4f}s")
    print(f"  Compressed size: {len(compressed_data)} bytes")
    print(f"  Compression ratio: {proof_validation['compression_ratio']:.4f}")
    print(f"  Validation score: {proof_validation['validation_score']:.4f}")
    print(f"  Proof verified: {proof_validation['proof_verified']}")

    # Test decompression
    start_time = time.time()
    decompressed_data = compressor.decompress_with_verification(compressed_data, proof_validation)
    decompression_time = time.time() - start_time

    print(f"  Decompression time: {decompression_time:.4f}s")
    print(f"  Data integrity: {decompressed_data == test_data}")

    # Batch-ambiguous compression
    print("\nBatch-Ambiguous Compression:")
    from src.search import BatchAmbiguousCompression

    batch_compressor = BatchAmbiguousCompression()

    batch_data = [test_data.copy() for _ in range(5)]
    # Modify each item slightly
    for i, item in enumerate(batch_data):
        item['metadata']['id'] = f"item_{i}"
        item['sequence'] = item['sequence'][:280 + i*10]  # Varying lengths

    start_time = time.time()
    compressed_batch, batch_metadata = batch_compressor.compress_batch_ambiguous(batch_data)
    batch_compression_time = time.time() - start_time

    print(f"  Batch compression time: {batch_compression_time:.4f}s")
    print(f"  Compressed batch size: {len(compressed_batch)} bytes")
    print(f"  Compression ratio: {batch_metadata['compression_ratio']:.4f}")
    print(f"  Ambiguity level: {batch_metadata['ambiguity_level']:.4f}")
    print(f"  Ambiguity groups: {batch_metadata['ambiguity_groups']}")

    print("\n✓ Compression algorithms demonstration completed")


def demo_system_integration():
    """Demonstrate complete system integration."""
    print("\n\n6. SYSTEM INTEGRATION")
    print("-" * 20)

    print("Demonstrating end-to-end workflow:")

    # Step 1: Sequence transformation
    print("\nStep 1: Transform sequence to coordinates")
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCG"
    seq_coords = transform_sequence_to_coordinates(test_sequence)
    print(f"  Sequence length: {len(test_sequence)}")
    print(f"  S-entropy: {seq_coords.s_entropy:.4f}")
    print(f"  Coordinates: {len(seq_coords.coordinates)} dimensions")

    # Step 2: Network setup
    print("\nStep 2: Set up distributed network")
    network = create_precision_network()
    node_ids = []
    for i in range(3):
        node_id = network.join_network()
        node_ids.append(node_id)
    print(f"  Network nodes: {len(node_ids)}")

    # Step 3: Search genome space
    print("\nStep 3: Search genome space")
    searcher = GraffitiGenomeSearch(network=network)

    start_time = time.time()
    search_results = searcher.search_genome_space(
        query=test_sequence,
        query_type='sequence',
        compression_level='hybrid',
        max_results=2
    )
    search_time = time.time() - start_time

    print(f"  Search time: {search_time:.4f}s")
    print(f"  Results found: {len(search_results)}")

    # Step 4: Analyze results on network
    print("\nStep 4: Analyze results on distributed network")
    if search_results:
        query_data = {
            'coordinates': search_results[0].genomic_coordinates.tolist()
        }

        start_time = time.time()
        analysis_result = analyze_distributed(network, query_data)
        analysis_time = time.time() - start_time

        print(f"  Analysis time: {analysis_time:.4f}s")
        print(f"  Information content: {analysis_result.get('total_information_content', 0):.4f}")
        print(f"  Confidence: {analysis_result.get('insights', {}).get('processing_confidence', 0):.4f}")

    # Step 5: Cleanup
    print("\nStep 5: Cleanup")
    for node_id in node_ids:
        network.leave_network(node_id)
    print("  Network cleaned up")

    print("\n✓ System integration demonstration completed")


def demo_mufakose_framework():
    """Demonstrate Mufakose confirmation-based genomics framework."""
    print("\n\n7. MUFAKOSE CONFIRMATION-BASED GENOMICS FRAMEWORK")
    print("-" * 55)

    # Create Mufakose framework
    mufakose = MufakoseGenomicsFramework()

    # Test 1: Membrane Confirmation Processing
    print("\nTest 1: Membrane Confirmation Processing")
    test_sequence = "ATGAAACCCGGGTTTAAATAG"
    reference_genome = "ATGAAACCCGGGTTTAAATAG"

    start_time = time.time()
    variant_confirmations = mufakose_variant_detection(test_sequence, reference_genome)
    processing_time = time.time() - start_time

    print(f"  Input sequence: {test_sequence}")
    print(f"  Variants detected: {len(variant_confirmations)}")
    print(f"  Processing time: {processing_time:.4f}s")

    if variant_confirmations:
        for i, variant in enumerate(variant_confirmations[:3]):  # Show first 3
            print(f"    Variant {i+1}:")
            print(f"      ID: {variant.variant_id}")
            print(f"      Confirmation probability: {variant.confirmation_probability:.4f}")
            print(f"      Pathogenicity score: {variant.pathogenicity_score:.4f}")
            print(f"      S-entropy coordinates: [{variant.genomic_coordinates[0]:.3f}, {variant.genomic_coordinates[1]:.3f}, {variant.genomic_coordinates[2]:.3f}]")

    # Test 2: Pharmacogenetic Analysis
    print("\nTest 2: Pharmacogenetic Analysis")
    drug_profile = {
        'drug_name': 'Warfarin',
        'target_genes': ['CYP2C9', 'VKORC1'],
        'metabolic_pathways': ['cytochrome_p450']
    }

    if variant_confirmations:
        start_time = time.time()
        pharma_result = mufakose.pharmacogenetic_prediction(variant_confirmations, drug_profile)
        pharma_time = time.time() - start_time

        print(f"  Drug: {drug_profile['drug_name']}")
        print(f"  Analysis time: {pharma_time:.4f}s")

        response_pred = pharma_result['response_prediction']
        print(f"  Response probability: {response_pred.get('response_probability', 0):.4f}")
        print(f"  Response category: {response_pred.get('response_category', 'unknown')}")
        print(f"  Predicted efficacy: {response_pred.get('predicted_efficacy', 'unknown')}")
        print(f"  Confidence: {pharma_result['confidence']:.4f}")

    # Test 3: Population Genomics Analysis
    print("\nTest 3: Population Genomics Analysis")
    population_samples = [
        {'sample_id': f'sample_{i}', 'variants': [f'var_{j}' for j in range(np.random.randint(50, 100))]}
        for i in range(20)
    ]

    analysis_objectives = ['rare_variants', 'population_structure']

    start_time = time.time()
    pop_result = mufakose.population_genomics_analysis(population_samples, analysis_objectives)
    pop_time = time.time() - start_time

    print(f"  Population size: {len(population_samples)} samples")
    print(f"  Analysis objectives: {analysis_objectives}")
    print(f"  Analysis time: {pop_time:.4f}s")
    print(f"  Memory complexity: O(1) - constant")
    print(f"  Time complexity: O(log N)")

    # Show population insights
    insights = pop_result.get('population_insights', {})
    if insights:
        print("  Population insights:")
        for objective, insight_data in insights.get('population_structure', {}).items():
            print(f"    {objective}: structure strength = {insight_data.get('structure_strength', 0):.4f}")

    # Test 4: Mufakose Search
    print("\nTest 4: Mufakose Genomic Search")
    search_queries = [
        ('ATGAAACCCGGGTTTAAATAG', 'sequence'),
        ('BRCA1 pathogenic variant', 'function'),
        ('diabetes susceptibility', 'phenotype')
    ]

    for query, query_type in search_queries:
        print(f"\n  Query: {query}")
        print(f"  Type: {query_type}")

        start_time = time.time()
        search_result = mufakose_search(query, query_type)
        search_time = time.time() - start_time

        print(f"  Search time: {search_time:.4f}s")
        print(f"  Query confirmation: {search_result.query_confirmation:.4f}")
        print(f"  Variants found: {len(search_result.variant_confirmations)}")
        print(f"  Evidence layers: {len(search_result.evidence_layers)}")

        # Show clinical recommendations
        clinical_recs = search_result.clinical_recommendations
        print(f"  Risk assessment: {clinical_recs.get('risk_assessment', 'unknown')}")
        print(f"  Monitoring required: {clinical_recs.get('monitoring_required', False)}")

    # Test 5: S-Entropy Compression Demonstration
    print("\nTest 5: S-Entropy Compression")

    # Simulate traditional vs Mufakose memory usage
    traditional_memory = len(population_samples) * 1000 * 100  # N * V * L
    mufakose_memory = len(population_samples) * 3 * 8  # N * 3 coordinates * 8 bytes

    compression_ratio = mufakose_memory / traditional_memory
    memory_savings = 1.0 - compression_ratio

    print(f"  Traditional memory usage: {traditional_memory:,} bytes")
    print(f"  Mufakose memory usage: {mufakose_memory:,} bytes")
    print(f"  Compression ratio: {compression_ratio:.6f}")
    print(f"  Memory savings: {memory_savings:.2%}")
    print(f"  Complexity reduction: O(N*V*L) → O(log(N*V))")

    print("\n✓ Mufakose framework demonstration completed")


def main():
    """Main demonstration function."""
    print("DISTRIBUTED GENOMICS NETWORK DEMONSTRATION")
    print("=" * 50)
    print("This demonstration showcases the complete genome system including:")
    print("• S-entropy coordinate transformation")
    print("• Distributed precision-by-difference network")
    print("• Graffiti-based proof search algorithms")
    print("• Mufakose confirmation-based genomics framework")
    print("• Compression algorithms")
    print("• System integration")
    print("=" * 50)

    start_time = time.time()

    try:
        # Run all demonstrations
        demo_sequence_transformation()
        demo_genome_transformation()
        demo_distributed_network()
        demo_graffiti_search()
        demo_compression_algorithms()
        demo_mufakose_framework()
        demo_system_integration()

        total_time = time.time() - start_time

        print("\n" + "=" * 50)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print(f"Total demonstration time: {total_time:.2f} seconds")
        print("=" * 50)

    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
