#!/usr/bin/env python3
"""
Βιολογική αξιολόγηση για remote homologs.
Χρήση UniProt annotations για ανάλυση λειτουργικής ομοιότητας.
"""

import requests
import time
from Bio import SeqIO
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiologicalAnalyzer:
    """Κλάση για βιολογική ανάλυση πρωτεϊνών."""
    
    def __init__(self):
        self.uniprot_base = "https://rest.uniprot.org/uniprotkb"
        self.cache = {}
        
    def fetch_uniprot_annotation(self, protein_id):
        """
        Ανάκτηση annotations από UniProt.
        
        Args:
            protein_id: UniProt ID
            
        Returns:
            dict: Annotations
        """
        # Έλεγχος cache
        if protein_id in self.cache:
            return self.cache[protein_id]
        
        try:
            # Extract clean UniProt ID
            clean_id = protein_id.split('|')[1] if '|' in protein_id else protein_id
            
            # API request
            url = f"{self.uniprot_base}/{clean_id}.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant information
                annotation = {
                    'id': clean_id,
                    'protein_name': self._get_protein_name(data),
                    'organism': self._get_organism(data),
                    'function': self._get_function(data),
                    'go_terms': self._get_go_terms(data),
                    'pfam_domains': self._get_pfam_domains(data),
                    'ec_numbers': self._get_ec_numbers(data),
                    'keywords': self._get_keywords(data)
                }
                
                self.cache[protein_id] = annotation
                return annotation
            else:
                logger.warning(f"UniProt API error για {protein_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {protein_id}: {e}")
            return None
        
        finally:
            # Rate limiting
            time.sleep(0.2)
    
    def _get_protein_name(self, data):
        """Extract protein name."""
        try:
            return data['proteinDescription']['recommendedName']['fullName']['value']
        except:
            try:
                return data['proteinDescription']['submissionNames'][0]['fullName']['value']
            except:
                return "Unknown"
    
    def _get_organism(self, data):
        """Extract organism."""
        try:
            return data['organism']['scientificName']
        except:
            return "Unknown"
    
    def _get_function(self, data):
        """Extract function description."""
        try:
            comments = data.get('comments', [])
            for comment in comments:
                if comment['commentType'] == 'FUNCTION':
                    return comment['texts'][0]['value']
        except:
            pass
        return "No function description"
    
    def _get_go_terms(self, data):
        """Extract GO terms."""
        go_terms = {
            'molecular_function': [],
            'biological_process': [],
            'cellular_component': []
        }
        
        try:
            for ref in data.get('uniProtKBCrossReferences', []):
                if ref['database'] == 'GO':
                    go_id = ref['id']
                    properties = {p['key']: p['value'] for p in ref.get('properties', [])}
                    go_type = properties.get('GoTerm', '').split(':')[0]
                    go_term = properties.get('GoTerm', '')
                    
                    if 'F:' in go_type:
                        go_terms['molecular_function'].append(go_term)
                    elif 'P:' in go_type:
                        go_terms['biological_process'].append(go_term)
                    elif 'C:' in go_type:
                        go_terms['cellular_component'].append(go_term)
        except:
            pass
        
        return go_terms
    
    def _get_pfam_domains(self, data):
        """Extract Pfam domains."""
        pfam_domains = []
        
        try:
            for ref in data.get('uniProtKBCrossReferences', []):
                if ref['database'] == 'Pfam':
                    pfam_id = ref['id']
                    properties = {p['key']: p['value'] for p in ref.get('properties', [])}
                    entry_name = properties.get('EntryName', '')
                    pfam_domains.append(f"{pfam_id} ({entry_name})")
        except:
            pass
        
        return pfam_domains
    
    def _get_ec_numbers(self, data):
        """Extract EC numbers."""
        ec_numbers = []
        
        try:
            if 'proteinDescription' in data:
                if 'ecNumbers' in data['proteinDescription'].get('recommendedName', {}):
                    ec_numbers = [ec['value'] for ec in data['proteinDescription']['recommendedName']['ecNumbers']]
        except:
            pass
        
        return ec_numbers
    
    def _get_keywords(self, data):
        """Extract keywords."""
        keywords = []
        
        try:
            for kw in data.get('keywords', []):
                keywords.append(kw['name'])
        except:
            pass
        
        return keywords
    
    def compare_proteins(self, protein1_id, protein2_id):
        """
        Σύγκριση δύο πρωτεϊνών για βιολογική ομοιότητα.
        
        Returns:
            dict: Comparison results
        """
        ann1 = self.fetch_uniprot_annotation(protein1_id)
        ann2 = self.fetch_uniprot_annotation(protein2_id)
        
        if not ann1 or not ann2:
            return None
        
        # Σύγκριση
        comparison = {
            'protein1': ann1,
            'protein2': ann2,
            'similarity': {}
        }
        
        # Common Pfam domains
        pfam1 = set(ann1['pfam_domains'])
        pfam2 = set(ann2['pfam_domains'])
        common_pfam = pfam1.intersection(pfam2)
        comparison['similarity']['common_pfam'] = list(common_pfam)
        comparison['similarity']['pfam_jaccard'] = self._jaccard(pfam1, pfam2)
        
        # Common GO terms
        go1_all = set(ann1['go_terms']['molecular_function'] + 
                     ann1['go_terms']['biological_process'] + 
                     ann1['go_terms']['cellular_component'])
        go2_all = set(ann2['go_terms']['molecular_function'] + 
                     ann2['go_terms']['biological_process'] + 
                     ann2['go_terms']['cellular_component'])
        common_go = go1_all.intersection(go2_all)
        comparison['similarity']['common_go'] = list(common_go)
        comparison['similarity']['go_jaccard'] = self._jaccard(go1_all, go2_all)
        
        # Common EC numbers
        ec1 = set(ann1['ec_numbers'])
        ec2 = set(ann2['ec_numbers'])
        common_ec = ec1.intersection(ec2)
        comparison['similarity']['common_ec'] = list(common_ec)
        
        # Common keywords
        kw1 = set(ann1['keywords'])
        kw2 = set(ann2['keywords'])
        common_kw = kw1.intersection(kw2)
        comparison['similarity']['common_keywords'] = list(common_kw)
        comparison['similarity']['keyword_jaccard'] = self._jaccard(kw1, kw2)
        
        # Overall similarity score
        comparison['similarity']['overall_score'] = (
            comparison['similarity']['pfam_jaccard'] * 0.4 +
            comparison['similarity']['go_jaccard'] * 0.4 +
            comparison['similarity']['keyword_jaccard'] * 0.2
        )
        
        return comparison
    
    def _jaccard(self, set1, set2):
        """Jaccard similarity."""
        if not set1 and not set2:
            return 0.0
        union = set1.union(set2)
        if not union:
            return 0.0
        intersection = set1.intersection(set2)
        return len(intersection) / len(union)
    
    def identify_remote_homologs(self, query_id, neighbors, blast_results, 
                                 l2_threshold=0.3, identity_threshold=30):
        """
        Εντοπισμός remote homologs.
        
        Args:
            query_id: Query protein ID
            neighbors: [(neighbor_id, l2_distance), ...]
            blast_results: [(hit_id, identity, e_value), ...]
            l2_threshold: Μέγιστη L2 απόσταση
            identity_threshold: Μέγιστο BLAST identity
            
        Returns:
            list: Candidate remote homologs με ανάλυση
        """
        blast_dict = {hit[0]: hit[1] for hit in blast_results}
        
        candidates = []
        
        for neighbor_id, l2_dist in neighbors:
            blast_identity = blast_dict.get(neighbor_id, 0.0)
            
            # Κριτήρια για remote homolog
            if l2_dist < l2_threshold and blast_identity < identity_threshold:
                # Βιολογική ανάλυση
                comparison = self.compare_proteins(query_id, neighbor_id)
                
                if comparison:
                    evidence = []
                    
                    # Έλεγχος evidence
                    if comparison['similarity']['common_pfam']:
                        evidence.append(f"Common Pfam: {', '.join(comparison['similarity']['common_pfam'][:3])}")
                    
                    if comparison['similarity']['common_ec']:
                        evidence.append(f"Common EC: {', '.join(comparison['similarity']['common_ec'])}")
                    
                    if comparison['similarity']['common_go']:
                        evidence.append(f"Common GO: {len(comparison['similarity']['common_go'])} terms")
                    
                    if comparison['similarity']['overall_score'] > 0.3:
                        candidates.append({
                            'neighbor_id': neighbor_id,
                            'l2_distance': l2_dist,
                            'blast_identity': blast_identity,
                            'similarity_score': comparison['similarity']['overall_score'],
                            'evidence': evidence,
                            'comparison': comparison
                        })
        
        # Sort by similarity score
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        return candidates
    
    def format_remote_homolog_report(self, query_id, candidates):
        """
        Δημιουργία αναφοράς για remote homologs.
        
        Returns:
            str: Formatted report
        """
        report = []
        report.append(f"\n{'='*80}")
        report.append(f"REMOTE HOMOLOG ANALYSIS για Query: {query_id}")
        report.append(f"{'='*80}\n")
        
        if not candidates:
            report.append("Δεν βρέθηκαν candidate remote homologs.")
            return "\n".join(report)
        
        for i, candidate in enumerate(candidates, 1):
            report.append(f"\nCandidate #{i}: {candidate['neighbor_id']}")
            report.append("-" * 80)
            report.append(f"L2 Distance: {candidate['l2_distance']:.4f}")
            report.append(f"BLAST Identity: {candidate['blast_identity']:.1f}% (Twilight Zone)")
            report.append(f"Similarity Score: {candidate['similarity_score']:.3f}")
            report.append(f"\nBiological Evidence:")
            
            if candidate['evidence']:
                for evidence in candidate['evidence']:
                    report.append(f"  • {evidence}")
            else:
                report.append("  • No strong evidence (potential false positive)")
            
            # Detailed comparison
            comp = candidate['comparison']
            report.append(f"\nDetailed Comparison:")
            report.append(f"  Query: {comp['protein1']['protein_name']} ({comp['protein1']['organism']})")
            report.append(f"  Neighbor: {comp['protein2']['protein_name']} ({comp['protein2']['organism']})")
            
            if comp['similarity']['common_pfam']:
                report.append(f"  Pfam Jaccard: {comp['similarity']['pfam_jaccard']:.3f}")
            if comp['similarity']['common_go']:
                report.append(f"  GO Jaccard: {comp['similarity']['go_jaccard']:.3f}")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Demo usage."""
    analyzer = BiologicalAnalyzer()
    
    # Example: Αναζήτηση annotation για μία πρωτεΐνη
    protein_id = "P12345"  # Replace with actual UniProt ID
    annotation = analyzer.fetch_uniprot_annotation(protein_id)
    
    if annotation:
        print(f"Protein: {annotation['protein_name']}")
        print(f"Organism: {annotation['organism']}")
        print(f"Pfam domains: {annotation['pfam_domains']}")
        print(f"GO terms: {len(annotation['go_terms']['molecular_function'])} MF, "
              f"{len(annotation['go_terms']['biological_process'])} BP, "
              f"{len(annotation['go_terms']['cellular_component'])} CC")


if __name__ == '__main__':
    main()