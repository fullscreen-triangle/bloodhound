from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from cyvcf2 import VCF, Variant
from pysam import VariantFile, TabixFile
import allel
from collections import defaultdict

class VCFProcessor:
    """Process and analyze VCF files"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vcf_readers = {}
        
    def load_vcf(self, file_path: Path) -> Dict[str, Any]:
        """Load and validate VCF file"""
        try:
            vcf = VCF(str(file_path))
            self.vcf_readers[str(file_path)] = vcf
            
            # Get basic VCF stats
            return {
                'samples': vcf.samples,
                'variants': len(vcf),
                'contigs': vcf.seqnames,
                'filters': vcf.filters,
                'formats': vcf.formats,
                'infos': vcf.infos
            }
            
        except Exception as e:
            logging.error(f"VCF loading error: {str(e)}")
            raise
            
    def analyze_variants(self, 
                        vcf_path: Path,
                        filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze variants in VCF file"""
        try:
            vcf = self.vcf_readers.get(str(vcf_path)) or VCF(str(vcf_path))
            
            # Initialize counters
            variant_types = defaultdict(int)
            quality_scores = []
            depth_values = []
            
            # Process variants
            for variant in vcf:
                if self._passes_filters(variant, filters):
                    # Count variant types
                    variant_types[self._get_variant_type(variant)] += 1
                    
                    # Collect quality metrics
                    if variant.QUAL:
                        quality_scores.append(variant.QUAL)
                    
                    # Get depth if available
                    depth = variant.format('DP')
                    if depth is not None:
                        depth_values.extend(depth.flatten())
            
            return {
                'variant_counts': dict(variant_types),
                'quality_stats': self._calculate_stats(quality_scores),
                'depth_stats': self._calculate_stats(depth_values),
                'total_variants': sum(variant_types.values())
            }
            
        except Exception as e:
            logging.error(f"Variant analysis error: {str(e)}")
            raise
            
    def compare_samples(self,
                       vcf_path: Path,
                       sample_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Compare variants between sample pairs"""
        try:
            vcf = self.vcf_readers.get(str(vcf_path)) or VCF(str(vcf_path))
            comparisons = []
            
            for sample1, sample2 in sample_pairs:
                # Get sample indices
                idx1 = vcf.samples.index(sample1)
                idx2 = vcf.samples.index(sample2)
                
                shared_variants = 0
                unique_sample1 = 0
                unique_sample2 = 0
                
                for variant in vcf:
                    gts = variant.genotypes
                    has_var1 = not all(gt[0] == 0 for gt in gts[idx1])
                    has_var2 = not all(gt[0] == 0 for gt in gts[idx2])
                    
                    if has_var1 and has_var2:
                        shared_variants += 1
                    elif has_var1:
                        unique_sample1 += 1
                    elif has_var2:
                        unique_sample2 += 1
                
                comparisons.append({
                    'sample1': sample1,
                    'sample2': sample2,
                    'shared_variants': shared_variants,
                    'unique_sample1': unique_sample1,
                    'unique_sample2': unique_sample2
                })
                
            return comparisons
            
        except Exception as e:
            logging.error(f"Sample comparison error: {str(e)}")
            raise
            
    def extract_genotypes(self,
                         vcf_path: Path,
                         samples: Optional[List[str]] = None,
                         region: Optional[str] = None) -> pd.DataFrame:
        """Extract genotype data for specified samples and region"""
        try:
            vcf = self.vcf_readers.get(str(vcf_path)) or VCF(str(vcf_path))
            
            # Filter samples if specified
            sample_indices = [vcf.samples.index(s) for s in (samples or vcf.samples)]
            
            variants = []
            for variant in (vcf(region) if region else vcf):
                var_data = {
                    'CHROM': variant.CHROM,
                    'POS': variant.POS,
                    'REF': variant.REF,
                    'ALT': ','.join(str(a) for a in variant.ALT),
                    'QUAL': variant.QUAL,
                    'FILTER': ','.join(variant.FILTER) if variant.FILTER else 'PASS'
                }
                
                # Add genotypes for selected samples
                gts = variant.genotypes
                for idx, sample in zip(sample_indices, samples or vcf.samples):
                    gt = gts[idx]
                    var_data[f'{sample}_GT'] = f'{gt[0]}/{gt[1]}'
                    
                variants.append(var_data)
                
            return pd.DataFrame(variants)
            
        except Exception as e:
            logging.error(f"Genotype extraction error: {str(e)}")
            raise
    
    def _passes_filters(self, 
                       variant: Variant,
                       filters: Optional[Dict[str, Any]] = None) -> bool:
        """Check if variant passes specified filters"""
        if not filters:
            return True
            
        try:
            if 'quality' in filters and variant.QUAL < filters['quality']:
                return False
                
            if 'depth' in filters:
                depth = variant.format('DP')
                if depth is None or np.mean(depth) < filters['depth']:
                    return False
                    
            if 'maf' in filters:
                af = variant.aaf
                if af < filters['maf'] or af > (1 - filters['maf']):
                    return False
                    
            return True
            
        except Exception:
            return False
    
    def _get_variant_type(self, variant: Variant) -> str:
        """Determine variant type"""
        ref_len = len(variant.REF)
        alt_lens = [len(alt) for alt in variant.ALT]
        
        if ref_len == 1 and all(l == 1 for l in alt_lens):
            return 'SNV'
        elif ref_len > 1 and all(l == 1 for l in alt_lens):
            return 'DELETION'
        elif ref_len == 1 and any(l > 1 for l in alt_lens):
            return 'INSERTION'
        else:
            return 'COMPLEX'
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics"""
        if not values:
            return {}
            
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
