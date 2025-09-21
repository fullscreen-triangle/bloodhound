"""
Graffiti-Based Genomic Search Implementation

Implements the proof-based search algorithm using LLM-assisted genome searching
with compression algorithms for efficient genomic space navigation.
"""

import numpy as np
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import asyncio
import requests
from datetime import datetime

from ..genome.genome_transform import GenomeCoordinates, GenomeTransformer
from ..sequence.s_entropy_transform import SEntropyCoordinates, SequenceTransformer
from ..network.network import PrecisionByDifferenceNetwork


@dataclass
class SearchQuery:
    """Represents a genomic search query."""
    query_id: str
    query_text: str
    query_type: str  # 'sequence', 'genome', 'phenotype', 'function'
    search_coordinates: np.ndarray
    compression_level: str  # 'proof_validated', 'batch_ambiguous', 'hybrid'
    llm_context: Dict[str, Any] = field(default_factory=dict)
    search_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a search result from the genomic space."""
    result_id: str
    relevance_score: float
    genomic_coordinates: np.ndarray
    genomic_data: Dict[str, Any]
    proof_validation: Dict[str, Any]
    compression_metrics: Dict[str, float]
    llm_reasoning: str
    confidence_level: float


@dataclass
class CompressionState:
    """State for compression algorithms."""
    proof_validated_state: Dict[str, Any] = field(default_factory=dict)
    batch_ambiguous_state: Dict[str, Any] = field(default_factory=dict)
    compression_history: List[Dict[str, Any]] = field(default_factory=list)


class ProofValidatedCompression:
    """
    Implements proof-validated compression for genomic data.

    Based on the theoretical framework from docs/garden/proof-validated-compression.tex
    """

    def __init__(self, validation_threshold: float = 0.95):
        self.validation_threshold = validation_threshold
        self.proof_cache = {}

    def compress_with_proof(self, genomic_data: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress genomic data with mathematical proof validation.

        Args:
            genomic_data: Genomic data to compress

        Returns:
            Tuple of (compressed_data, proof_validation)
        """
        # Generate data hash for proof validation
        data_hash = self._generate_data_hash(genomic_data)

        # Create mathematical proof of data integrity
        proof = self._generate_mathematical_proof(genomic_data, data_hash)

        # Compress data using proof-guided compression
        compressed_data = self._proof_guided_compression(genomic_data, proof)

        # Validate compression integrity
        validation_result = self._validate_compression_proof(compressed_data, proof, data_hash)

        proof_validation = {
            'original_hash': data_hash,
            'proof_hash': proof['proof_hash'],
            'validation_score': validation_result['score'],
            'compression_ratio': len(compressed_data) / len(json.dumps(genomic_data).encode()),
            'proof_verified': validation_result['score'] >= self.validation_threshold
        }

        return compressed_data, proof_validation

    def decompress_with_verification(self, compressed_data: bytes,
                                   proof_validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompress data with proof verification.

        Args:
            compressed_data: Compressed genomic data
            proof_validation: Proof validation metadata

        Returns:
            Decompressed genomic data
        """
        # Decompress data
        decompressed_data = self._proof_guided_decompression(compressed_data, proof_validation)

        # Verify data integrity using proof
        verification_hash = self._generate_data_hash(decompressed_data)

        if verification_hash != proof_validation['original_hash']:
            raise ValueError("Proof validation failed: Data integrity compromised")

        return decompressed_data

    def _generate_data_hash(self, data: Dict[str, Any]) -> str:
        """Generate cryptographic hash of genomic data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _generate_mathematical_proof(self, data: Dict[str, Any], data_hash: str) -> Dict[str, Any]:
        """Generate mathematical proof of data properties."""
        # Extract data properties for proof generation
        if 'sequence' in data:
            sequence = data['sequence']
            sequence_length = len(sequence)
            gc_content = sequence.count('G') + sequence.count('C')

            # Mathematical properties
            properties = {
                'length': sequence_length,
                'gc_count': gc_content,
                'at_count': sequence_length - gc_content,
                'entropy': self._calculate_sequence_entropy(sequence)
            }
        elif 'genome' in data:
            genome = data['genome']
            total_length = sum(len(seq) for seq in genome.values())
            chromosome_count = len(genome)

            properties = {
                'total_length': total_length,
                'chromosome_count': chromosome_count,
                'average_chromosome_length': total_length / chromosome_count if chromosome_count > 0 else 0
            }
        else:
            properties = {'data_size': len(str(data))}

        # Generate proof based on properties
        proof_components = []
        for key, value in properties.items():
            # Simple mathematical proof: property value modulo prime
            prime = 1009  # Large prime for proof generation
            proof_component = (hash(f"{key}_{value}") % prime)
            proof_components.append(proof_component)

        proof_value = sum(proof_components) % (prime * prime)

        proof = {
            'properties': properties,
            'proof_components': proof_components,
            'proof_value': proof_value,
            'proof_hash': hashlib.sha256(f"{data_hash}_{proof_value}".encode()).hexdigest()
        }

        return proof

    def _calculate_sequence_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of a sequence."""
        if not sequence:
            return 0.0

        # Count nucleotide frequencies
        counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for nuc in sequence:
            if nuc in counts:
                counts[nuc] += 1

        # Calculate entropy
        total = len(sequence)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    def _proof_guided_compression(self, data: Dict[str, Any], proof: Dict[str, Any]) -> bytes:
        """Compress data using proof-guided algorithm."""
        # Convert data to JSON string
        data_str = json.dumps(data, sort_keys=True)

        # Simple compression using proof properties
        # In a real implementation, this would use sophisticated compression
        compressed_str = self._simple_compress(data_str, proof['proof_value'])

        return compressed_str.encode()

    def _proof_guided_decompression(self, compressed_data: bytes,
                                  proof_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress data using proof validation."""
        compressed_str = compressed_data.decode()

        # Extract proof value from validation
        # In a real implementation, this would reconstruct the proof
        proof_value = hash(proof_validation['proof_hash']) % (1009 * 1009)

        # Decompress using proof
        decompressed_str = self._simple_decompress(compressed_str, proof_value)

        return json.loads(decompressed_str)

    def _simple_compress(self, data_str: str, proof_value: int) -> str:
        """Simple compression algorithm guided by proof value."""
        # XOR-based compression using proof value as key
        compressed_chars = []
        key = proof_value % 256

        for i, char in enumerate(data_str):
            compressed_char = chr(ord(char) ^ (key + i) % 256)
            compressed_chars.append(compressed_char)

        return ''.join(compressed_chars)

    def _simple_decompress(self, compressed_str: str, proof_value: int) -> str:
        """Simple decompression algorithm."""
        # Reverse XOR operation
        decompressed_chars = []
        key = proof_value % 256

        for i, char in enumerate(compressed_str):
            decompressed_char = chr(ord(char) ^ (key + i) % 256)
            decompressed_chars.append(decompressed_char)

        return ''.join(decompressed_chars)

    def _validate_compression_proof(self, compressed_data: bytes,
                                  proof: Dict[str, Any],
                                  original_hash: str) -> Dict[str, Any]:
        """Validate compression using mathematical proof."""
        # Decompress and verify
        try:
            decompressed = self._proof_guided_decompression(compressed_data, {'proof_hash': proof['proof_hash']})
            verification_hash = self._generate_data_hash(decompressed)

            # Calculate validation score
            hash_match = verification_hash == original_hash
            proof_integrity = self._verify_proof_integrity(proof)

            score = (0.7 if hash_match else 0.0) + (0.3 if proof_integrity else 0.0)

            return {
                'score': score,
                'hash_match': hash_match,
                'proof_integrity': proof_integrity
            }
        except Exception:
            return {'score': 0.0, 'hash_match': False, 'proof_integrity': False}

    def _verify_proof_integrity(self, proof: Dict[str, Any]) -> bool:
        """Verify mathematical proof integrity."""
        # Recalculate proof components
        properties = proof['properties']
        recalculated_components = []

        for key, value in properties.items():
            prime = 1009
            proof_component = (hash(f"{key}_{value}") % prime)
            recalculated_components.append(proof_component)

        recalculated_value = sum(recalculated_components) % (prime * prime)

        return recalculated_value == proof['proof_value']


class BatchAmbiguousCompression:
    """
    Implements batch-ambiguous compression for genomic data.

    Based on the theoretical framework from docs/garden/batch-ambigious-compression.tex
    """

    def __init__(self, batch_size: int = 100, ambiguity_threshold: float = 0.1):
        self.batch_size = batch_size
        self.ambiguity_threshold = ambiguity_threshold
        self.batch_cache = {}

    def compress_batch_ambiguous(self, genomic_batch: List[Dict[str, Any]]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress a batch of genomic data using ambiguous compression.

        Args:
            genomic_batch: List of genomic data items

        Returns:
            Tuple of (compressed_batch, compression_metadata)
        """
        # Create ambiguity groups
        ambiguity_groups = self._create_ambiguity_groups(genomic_batch)

        # Compress each group with controlled ambiguity
        compressed_groups = []
        for group in ambiguity_groups:
            compressed_group = self._compress_ambiguous_group(group)
            compressed_groups.append(compressed_group)

        # Serialize compressed groups
        compressed_data = json.dumps(compressed_groups).encode()

        # Generate compression metadata
        metadata = {
            'original_batch_size': len(genomic_batch),
            'ambiguity_groups': len(ambiguity_groups),
            'compression_ratio': len(compressed_data) / len(json.dumps(genomic_batch).encode()),
            'ambiguity_level': self._calculate_ambiguity_level(ambiguity_groups),
            'batch_id': hashlib.sha256(compressed_data).hexdigest()[:16]
        }

        return compressed_data, metadata

    def decompress_batch_ambiguous(self, compressed_data: bytes,
                                 metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompress batch-ambiguous data.

        Args:
            compressed_data: Compressed batch data
            metadata: Compression metadata

        Returns:
            Decompressed genomic batch
        """
        # Deserialize compressed groups
        compressed_groups = json.loads(compressed_data.decode())

        # Decompress each group
        decompressed_batch = []
        for compressed_group in compressed_groups:
            decompressed_group = self._decompress_ambiguous_group(compressed_group)
            decompressed_batch.extend(decompressed_group)

        return decompressed_batch

    def _create_ambiguity_groups(self, genomic_batch: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create ambiguity groups from genomic batch."""
        groups = []
        current_group = []

        for item in genomic_batch:
            # Add item to current group
            current_group.append(item)

            # Check if group should be finalized
            if (len(current_group) >= self.batch_size or
                self._calculate_group_ambiguity(current_group) > self.ambiguity_threshold):
                groups.append(current_group)
                current_group = []

        # Add remaining items
        if current_group:
            groups.append(current_group)

        return groups

    def _calculate_group_ambiguity(self, group: List[Dict[str, Any]]) -> float:
        """Calculate ambiguity level of a group."""
        if len(group) <= 1:
            return 0.0

        # Calculate similarity between items
        similarities = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                similarity = self._calculate_item_similarity(group[i], group[j])
                similarities.append(similarity)

        # Ambiguity is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        ambiguity = 1.0 - avg_similarity

        return ambiguity

    def _calculate_item_similarity(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> float:
        """Calculate similarity between two genomic items."""
        # Simple similarity based on common keys and values
        keys1 = set(item1.keys())
        keys2 = set(item2.keys())

        common_keys = keys1.intersection(keys2)
        if not common_keys:
            return 0.0

        # Calculate value similarity for common keys
        similarities = []
        for key in common_keys:
            val1, val2 = item1[key], item2[key]

            if isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple)
                similarity = len(set(val1).intersection(set(val2))) / len(set(val1).union(set(val2)))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                similarity = 1.0 / (1.0 + abs(val1 - val2))
            else:
                # Exact match
                similarity = 1.0 if val1 == val2 else 0.0

            similarities.append(similarity)

        return np.mean(similarities)

    def _compress_ambiguous_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compress a group using ambiguous compression."""
        if not group:
            return {'type': 'empty', 'data': []}

        # Extract common patterns
        common_keys = set(group[0].keys())
        for item in group[1:]:
            common_keys = common_keys.intersection(set(item.keys()))

        # Create compressed representation
        compressed_group = {
            'type': 'ambiguous_group',
            'common_keys': list(common_keys),
            'group_size': len(group),
            'compressed_items': []
        }

        # Compress each item relative to common structure
        for item in group:
            compressed_item = {}
            for key in common_keys:
                compressed_item[key] = item[key]

            # Store unique keys separately
            unique_keys = set(item.keys()) - common_keys
            if unique_keys:
                compressed_item['_unique'] = {key: item[key] for key in unique_keys}

            compressed_group['compressed_items'].append(compressed_item)

        return compressed_group

    def _decompress_ambiguous_group(self, compressed_group: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompress an ambiguous group."""
        if compressed_group['type'] == 'empty':
            return []

        decompressed_items = []

        for compressed_item in compressed_group['compressed_items']:
            # Reconstruct item
            item = {}

            # Add common keys
            for key in compressed_group['common_keys']:
                if key in compressed_item:
                    item[key] = compressed_item[key]

            # Add unique keys
            if '_unique' in compressed_item:
                item.update(compressed_item['_unique'])

            decompressed_items.append(item)

        return decompressed_items

    def _calculate_ambiguity_level(self, ambiguity_groups: List[List[Dict[str, Any]]]) -> float:
        """Calculate overall ambiguity level of the batch."""
        if not ambiguity_groups:
            return 0.0

        group_ambiguities = [self._calculate_group_ambiguity(group) for group in ambiguity_groups]
        return np.mean(group_ambiguities)


class GraffitiGenomeSearch:
    """
    Implements Graffiti-based genomic search using LLM assistance.

    Integrates with the distributed genomics network and compression algorithms
    to provide proof-based search capabilities.
    """

    def __init__(self,
                 network: Optional[PrecisionByDifferenceNetwork] = None,
                 coordinate_dimensions: int = 128):
        self.network = network
        self.coordinate_dimensions = coordinate_dimensions

        # Initialize transformers
        self.genome_transformer = GenomeTransformer()
        self.sequence_transformer = SequenceTransformer()

        # Initialize compression algorithms
        self.proof_compression = ProofValidatedCompression()
        self.batch_compression = BatchAmbiguousCompression()

        # Search state
        self.search_history = []
        self.llm_context_cache = {}

    def search_genome_space(self,
                          query: str,
                          query_type: str = 'sequence',
                          compression_level: str = 'hybrid',
                          max_results: int = 10) -> List[SearchResult]:
        """
        Search the genomic space using Graffiti proof-based search.

        Args:
            query: Search query (sequence, description, etc.)
            query_type: Type of query ('sequence', 'genome', 'phenotype', 'function')
            compression_level: Compression algorithm to use
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects
        """
        # Create search query object
        search_query = self._create_search_query(query, query_type, compression_level)

        # Generate LLM context for search
        llm_context = self._generate_llm_context(search_query)
        search_query.llm_context = llm_context

        # Perform coordinate-based search
        coordinate_results = self._coordinate_search(search_query)

        # Apply compression-based filtering
        compressed_results = self._apply_compression_filtering(coordinate_results, search_query)

        # LLM-assisted result ranking
        ranked_results = self._llm_rank_results(compressed_results, search_query)

        # Generate final search results
        final_results = self._generate_search_results(ranked_results, search_query)

        # Store search in history
        self.search_history.append({
            'query': search_query,
            'results': final_results,
            'timestamp': datetime.now()
        })

        return final_results[:max_results]

    def _create_search_query(self, query: str, query_type: str, compression_level: str) -> SearchQuery:
        """Create a SearchQuery object from input parameters."""
        query_id = hashlib.sha256(f"{query}_{query_type}_{time.time()}".encode()).hexdigest()[:16]

        # Transform query to coordinates
        if query_type == 'sequence':
            # Direct sequence transformation
            seq_coords = self.sequence_transformer.transform_to_coordinates(query)
            search_coordinates = seq_coords.coordinates
        elif query_type == 'genome':
            # Assume query is a genome description or identifier
            # For demo, create synthetic coordinates
            search_coordinates = np.random.randn(self.coordinate_dimensions)
        else:
            # For phenotype/function queries, generate coordinates from text
            search_coordinates = self._text_to_coordinates(query)

        return SearchQuery(
            query_id=query_id,
            query_text=query,
            query_type=query_type,
            search_coordinates=search_coordinates,
            compression_level=compression_level,
            search_metadata={
                'timestamp': time.time(),
                'coordinate_magnitude': np.linalg.norm(search_coordinates)
            }
        )

    def _text_to_coordinates(self, text: str) -> np.ndarray:
        """Convert text query to coordinate representation."""
        # Simple text-to-coordinate transformation
        # In a real implementation, this would use NLP embeddings

        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Convert hash to coordinates
        coordinates = []
        for i in range(0, min(len(text_hash), self.coordinate_dimensions * 2), 2):
            hex_pair = text_hash[i:i+2]
            coord_value = int(hex_pair, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
            coordinates.append(coord_value)

        # Pad or truncate to desired dimensions
        while len(coordinates) < self.coordinate_dimensions:
            coordinates.append(0.0)

        return np.array(coordinates[:self.coordinate_dimensions])

    def _generate_llm_context(self, search_query: SearchQuery) -> Dict[str, Any]:
        """Generate LLM context for the search query."""
        # Simulate LLM context generation
        # In a real implementation, this would call an actual LLM API

        context = {
            'query_interpretation': f"Genomic search for: {search_query.query_text}",
            'search_strategy': self._determine_search_strategy(search_query),
            'expected_result_types': self._predict_result_types(search_query),
            'coordinate_analysis': {
                'magnitude': float(np.linalg.norm(search_query.search_coordinates)),
                'dominant_dimensions': np.argsort(np.abs(search_query.search_coordinates))[-5:].tolist()
            }
        }

        return context

    def _determine_search_strategy(self, search_query: SearchQuery) -> str:
        """Determine optimal search strategy based on query."""
        if search_query.query_type == 'sequence':
            return 'coordinate_similarity'
        elif search_query.query_type == 'genome':
            return 'hierarchical_coordinate_search'
        elif search_query.query_type == 'phenotype':
            return 'functional_coordinate_mapping'
        else:
            return 'general_coordinate_navigation'

    def _predict_result_types(self, search_query: SearchQuery) -> List[str]:
        """Predict types of results expected from the search."""
        if search_query.query_type == 'sequence':
            return ['similar_sequences', 'homologous_genes', 'functional_domains']
        elif search_query.query_type == 'genome':
            return ['related_genomes', 'phylogenetic_neighbors', 'functional_similarities']
        elif search_query.query_type == 'phenotype':
            return ['associated_genes', 'pathway_components', 'regulatory_elements']
        else:
            return ['general_matches', 'coordinate_neighbors']

    def _coordinate_search(self, search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform coordinate-based search in genomic space."""
        # If network is available, use distributed search
        if self.network:
            return self._distributed_coordinate_search(search_query)
        else:
            return self._local_coordinate_search(search_query)

    def _distributed_coordinate_search(self, search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform distributed coordinate search using the network."""
        # Create network query
        network_query = {
            'coordinates': search_query.search_coordinates.tolist(),
            'query_type': search_query.query_type,
            'instructions': {
                'search_radius': 1.0,
                'max_results': 50,
                'compression_level': search_query.compression_level
            }
        }

        # Process query on network
        try:
            from ..network.network import analyze_distributed
            network_results = analyze_distributed(self.network, network_query)

            # Convert network results to coordinate search results
            coordinate_results = [{
                'coordinates': search_query.search_coordinates,
                'similarity_score': network_results.get('insights', {}).get('similarity_assessment', 0.5),
                'source': 'distributed_network',
                'network_metadata': network_results
            }]

        except Exception as e:
            print(f"Distributed search failed: {e}")
            coordinate_results = self._local_coordinate_search(search_query)

        return coordinate_results

    def _local_coordinate_search(self, search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform local coordinate search."""
        # Generate synthetic search results for demonstration
        results = []

        # Create multiple result candidates
        for i in range(20):
            # Generate coordinates near the search coordinates
            noise_level = 0.1 + i * 0.05  # Increasing distance
            result_coords = search_query.search_coordinates + np.random.normal(0, noise_level,
                                                                             len(search_query.search_coordinates))

            # Calculate similarity score
            distance = np.linalg.norm(result_coords - search_query.search_coordinates)
            similarity_score = 1.0 / (1.0 + distance)

            result = {
                'coordinates': result_coords,
                'similarity_score': similarity_score,
                'source': 'local_search',
                'result_index': i
            }

            results.append(result)

        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return results

    def _apply_compression_filtering(self, coordinate_results: List[Dict[str, Any]],
                                   search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Apply compression-based filtering to search results."""
        filtered_results = []

        for result in coordinate_results:
            # Create genomic data for compression testing
            genomic_data = {
                'coordinates': result['coordinates'].tolist(),
                'query_type': search_query.query_type,
                'similarity_score': result['similarity_score']
            }

            # Apply compression based on level
            if search_query.compression_level in ['proof_validated', 'hybrid']:
                compressed_data, proof_validation = self.proof_compression.compress_with_proof(genomic_data)

                # Filter based on proof validation
                if proof_validation['proof_verified']:
                    result['proof_validation'] = proof_validation
                    result['compression_metrics'] = {
                        'proof_validated_ratio': proof_validation['compression_ratio'],
                        'validation_score': proof_validation['validation_score']
                    }
                    filtered_results.append(result)

            elif search_query.compression_level in ['batch_ambiguous', 'hybrid']:
                # For batch compression, group similar results
                batch_data = [genomic_data]
                compressed_batch, batch_metadata = self.batch_compression.compress_batch_ambiguous(batch_data)

                result['batch_compression'] = batch_metadata
                result['compression_metrics'] = {
                    'batch_compression_ratio': batch_metadata['compression_ratio'],
                    'ambiguity_level': batch_metadata['ambiguity_level']
                }
                filtered_results.append(result)

            else:
                # No compression filtering
                result['compression_metrics'] = {'compression_ratio': 1.0}
                filtered_results.append(result)

        return filtered_results

    def _llm_rank_results(self, compressed_results: List[Dict[str, Any]],
                         search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Use LLM assistance to rank search results."""
        # Simulate LLM-based ranking
        # In a real implementation, this would use an actual LLM API

        for result in compressed_results:
            # Generate LLM reasoning
            llm_reasoning = self._generate_llm_reasoning(result, search_query)
            result['llm_reasoning'] = llm_reasoning

            # Calculate LLM-adjusted score
            base_score = result['similarity_score']
            compression_bonus = result.get('compression_metrics', {}).get('validation_score', 0.5)

            # LLM confidence based on reasoning quality
            reasoning_quality = len(llm_reasoning.split()) / 50.0  # Simple metric
            llm_confidence = min(1.0, reasoning_quality)

            # Combined score
            result['llm_adjusted_score'] = (base_score * 0.6 +
                                          compression_bonus * 0.2 +
                                          llm_confidence * 0.2)
            result['llm_confidence'] = llm_confidence

        # Sort by LLM-adjusted score
        compressed_results.sort(key=lambda x: x['llm_adjusted_score'], reverse=True)

        return compressed_results

    def _generate_llm_reasoning(self, result: Dict[str, Any], search_query: SearchQuery) -> str:
        """Generate LLM reasoning for a search result."""
        # Simulate LLM reasoning generation
        similarity = result['similarity_score']

        if similarity > 0.8:
            reasoning = f"High similarity match for {search_query.query_type} query. "
        elif similarity > 0.5:
            reasoning = f"Moderate similarity match for {search_query.query_type} query. "
        else:
            reasoning = f"Low similarity match for {search_query.query_type} query. "

        # Add compression reasoning
        if 'proof_validation' in result:
            if result['proof_validation']['proof_verified']:
                reasoning += "Proof validation confirms data integrity. "
            else:
                reasoning += "Proof validation shows potential data issues. "

        if 'batch_compression' in result:
            ambiguity = result['batch_compression']['ambiguity_level']
            if ambiguity < 0.3:
                reasoning += "Low ambiguity indicates high confidence result. "
            else:
                reasoning += "Higher ambiguity suggests result requires careful interpretation. "

        # Add coordinate analysis
        coord_magnitude = np.linalg.norm(result['coordinates'])
        if coord_magnitude > 2.0:
            reasoning += "High coordinate magnitude suggests complex genomic features. "

        return reasoning

    def _generate_search_results(self, ranked_results: List[Dict[str, Any]],
                               search_query: SearchQuery) -> List[SearchResult]:
        """Generate final SearchResult objects."""
        search_results = []

        for i, result in enumerate(ranked_results):
            result_id = f"{search_query.query_id}_result_{i}"

            # Create genomic data representation
            genomic_data = {
                'coordinates': result['coordinates'].tolist(),
                'source': result.get('source', 'unknown'),
                'result_index': i,
                'search_metadata': result
            }

            # Extract validation and compression info
            proof_validation = result.get('proof_validation', {})
            compression_metrics = result.get('compression_metrics', {})

            search_result = SearchResult(
                result_id=result_id,
                relevance_score=result['llm_adjusted_score'],
                genomic_coordinates=result['coordinates'],
                genomic_data=genomic_data,
                proof_validation=proof_validation,
                compression_metrics=compression_metrics,
                llm_reasoning=result.get('llm_reasoning', ''),
                confidence_level=result.get('llm_confidence', 0.5)
            )

            search_results.append(search_result)

        return search_results

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get search history."""
        return self.search_history

    def clear_search_history(self):
        """Clear search history."""
        self.search_history = []


def search_genome_space(query: str,
                       query_type: str = 'sequence',
                       compression_level: str = 'hybrid',
                       network: Optional[PrecisionByDifferenceNetwork] = None,
                       max_results: int = 10) -> List[SearchResult]:
    """
    Convenience function to search the genomic space.

    Args:
        query: Search query
        query_type: Type of query
        compression_level: Compression algorithm to use
        network: Optional distributed network
        max_results: Maximum number of results

    Returns:
        List of SearchResult objects
    """
    searcher = GraffitiGenomeSearch(network=network)
    return searcher.search_genome_space(query, query_type, compression_level, max_results)


def compress_with_validation(genomic_data: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convenience function for proof-validated compression.

    Args:
        genomic_data: Genomic data to compress

    Returns:
        Tuple of (compressed_data, proof_validation)
    """
    compressor = ProofValidatedCompression()
    return compressor.compress_with_proof(genomic_data)
