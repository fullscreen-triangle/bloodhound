from typing import Dict, Any, List, Optional, Set
import logging
from pathlib import Path
import pandas as pd
import networkx as nx
from goatools import obo_parser
from goatools.go_enrichment import GOEnrichmentStudy
from scipy import stats

class PathwayAnalyzer:
    """Analyze biological pathways"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.go = self._load_go_ontology()
        self.pathways = self._load_pathway_database()
        
    def analyze_enrichment(self, 
                          gene_list: List[str],
                          background_genes: Optional[List[str]] = None,
                          p_value_cutoff: float = 0.05) -> List[Dict[str, Any]]:
        """Perform pathway enrichment analysis"""
        try:
            if background_genes is None:
                background_genes = self._get_default_background()
                
            # Create GO enrichment study
            go_study = GOEnrichmentStudy(
                background_genes,
                self.go,
                propagate_counts=True,
                alpha=p_value_cutoff
            )
            
            # Run enrichment analysis
            results = go_study.run_study(gene_list)
            
            # Format results
            enriched_pathways = []
            for r in results:
                if r.p_fdr_bh < p_value_cutoff:  # Use FDR-corrected p-value
                    enriched_pathways.append({
                        'id': r.GO,
                        'name': r.name,
                        'p_value': r.p_uncorrected,
                        'p_value_adjusted': r.p_fdr_bh,
                        'genes': r.study_items,
                        'category': r.NS
                    })
                    
            return enriched_pathways
            
        except Exception as e:
            logging.error(f"Enrichment analysis error: {str(e)}")
            raise
            
    def find_metabolic_pathways(self, 
                               compounds: List[str]) -> List[Dict[str, Any]]:
        """Find metabolic pathways containing compounds"""
        try:
            pathways = []
            compound_set = set(compounds)
            
            for pathway in self.pathways:
                pathway_compounds = set(pathway['compounds'])
                overlap = compound_set.intersection(pathway_compounds)
                
                if overlap:
                    # Calculate enrichment statistics
                    stats_result = self._calculate_pathway_stats(
                        overlap_size=len(overlap),
                        pathway_size=len(pathway_compounds),
                        total_compounds=len(compound_set)
                    )
                    
                    pathways.append({
                        'id': pathway['id'],
                        'name': pathway['name'],
                        'compounds': list(overlap),
                        'p_value': stats_result['p_value'],
                        'enrichment_score': stats_result['enrichment_score']
                    })
                    
            return sorted(pathways, key=lambda x: x['p_value'])
            
        except Exception as e:
            logging.error(f"Metabolic pathway analysis error: {str(e)}")
            raise
            
    def _load_go_ontology(self) -> Any:
        """Load Gene Ontology"""
        try:
            go_path = self.config.get('go_path', 'data/go.obo')
            return obo_parser.GODag(go_path)
        except Exception as e:
            logging.error(f"GO loading error: {str(e)}")
            raise
            
    def _load_pathway_database(self) -> List[Dict[str, Any]]:
        """Load pathway database"""
        try:
            pathway_file = self.config.get('pathway_file', 'data/pathways.json')
            return pd.read_json(pathway_file).to_dict('records')
        except Exception as e:
            logging.error(f"Pathway database loading error: {str(e)}")
            raise
            
    def _get_default_background(self) -> List[str]:
        """Get default background gene list"""
        try:
            background_file = self.config.get('background_file', 'data/background_genes.txt')
            with open(background_file) as f:
                return [line.strip() for line in f]
        except Exception as e:
            logging.error(f"Background loading error: {str(e)}")
            raise
            
    def _calculate_pathway_stats(self,
                               overlap_size: int,
                               pathway_size: int,
                               total_compounds: int) -> Dict[str, float]:
        """Calculate pathway statistics"""
        try:
            # Hypergeometric test
            p_value = stats.hypergeom.sf(
                overlap_size - 1,  # number of successes
                total_compounds,   # population size
                pathway_size,      # number of successes in population
                total_compounds    # number of draws
            )
            
            # Enrichment score
            enrichment_score = (overlap_size / pathway_size) * \
                             (overlap_size / total_compounds)
                             
            return {
                'p_value': p_value,
                'enrichment_score': enrichment_score
            }
            
        except Exception as e:
            logging.error(f"Statistics calculation error: {str(e)}")
            raise
