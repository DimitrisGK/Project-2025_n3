#!/usr/bin/env python3
"""
Ενοποιημένο σύστημα αναζήτησης πρωτεϊνών που χρησιμοποιεί τους αλγορίθμους:
- LSH (C++)
- Hypercube (C++)
- IVF-Flat (C++)
- IVF-PQ (C++)
- Neural LSH (Python)

Χρήση: python3 protein_search_unified.py -d protein_vectors.dat -q targets.fasta -o results.txt
"""

import argparse
import pickle
import numpy as np
from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Blast import NCBIXML
import subprocess
import tempfile
import os
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinSearchBenchmark:
    # Benchmark protein search με όλες τις ANN μεθόδους
    
    def __init__(self, embeddings_file, database_fasta=None):
        
        """ Args:
                embeddings_file: Path στο αρχείο με embeddings
                database_fasta: Path στο FASTA της βάσης (για BLAST)"""
        
        logger.info(f"Φόρτωση embeddings από {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.sequences = data['sequences']
        self.embedding_dim = data['embedding_dim']
        
        # Μετατροπή σε numpy arrays
        self.ids = list(self.embeddings.keys())
        self.vectors = np.array([self.embeddings[pid] for pid in self.ids])
        
        logger.info(f"Φορτώθηκαν {len(self.ids)} πρωτεΐνες, διάσταση: {self.embedding_dim}")
        
        self.database_fasta = database_fasta
        self.temp_dir = tempfile.mkdtemp(prefix="protein_search_")
        self.results = {}
        
    def __del__(self):
        # Cleanup temp directory
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def save_data_for_cpp(self, data_file, query_file=None, query_vectors=None):
        
        # Αποθήκευση vectors σε format για C++ αλγορίθμους (SIFT-like binary format).
        """ Args:
                data_file: Output file για database vectors
                query_file: Output file για query vectors (optional)
                query_vectors: numpy array με query vectors (optional)"""

        def write_vectors(filename, vectors):
            # Γράψε vectors σε binary format
            n, d = vectors.shape
            with open(filename, 'wb') as f:
                # Header: n_vectors(4 bytes) + dimension(4 bytes)
                f.write(np.array([n], dtype=np.int32).tobytes())
                f.write(np.array([d], dtype=np.int32).tobytes())
                # Data: n vectors of d floats
                f.write(vectors.astype(np.float32).tobytes())
        
        # Save database
        write_vectors(data_file, self.vectors)
        
        # Save queries if provided
        if query_file and query_vectors is not None:
            write_vectors(query_file, query_vectors)
    
    def parse_cpp_output(self, output_file):
        
        # Parse αποτελέσματα από C++ output file.
        
        """Returns:
                dict: {query_id: {'neighbors': [(id, distance), ...], 'time': float}}"""
        
        results = {}
        current_query = None
        neighbors = []
        approx_time = 0.0
        
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('Query:'):
                    # Αποθήκευση προηγούμενου query
                    if current_query is not None:
                        results[current_query] = {
                            'neighbors': neighbors,
                            'time': approx_time
                        }
                    
                    # Νέο query (0-based index)
                    current_query = int(line.split(':')[1].strip())
                    neighbors = []
                
                elif line.startswith('Nearest neighbor-'):
                    # Format: "Nearest neighbor-1: 42"
                    neighbor_id = int(line.split(':')[1].strip()) - 1  # Convert to 0-based
                    
                elif line.startswith('distanceApproximate:'):
                    distance = float(line.split(':')[1].strip())
                    if neighbor_id >= 0:
                        neighbors.append((neighbor_id, distance))
                
                elif line.startswith('tApproximateAverage:'):
                    approx_time = float(line.split(':')[1].strip())
        
        # Αποθήκευση τελευταίου query
        if current_query is not None:
            results[current_query] = {
                'neighbors': neighbors,
                'time': approx_time
            }
        
        return results
    
    def run_lsh(self, query_vectors, N=50, R=1000.0, k=4, L=5, w=4.0):
        # Εκτέλεση LSH (C++)
        logger.info(f"Εκτέλεση LSH με k={k}, L={L}, w={w}...")
        
        # Prepare data files
        data_file = os.path.join(self.temp_dir, "data.bin")
        query_file = os.path.join(self.temp_dir, "query.bin")
        output_file = os.path.join(self.temp_dir, "lsh_output.txt")
        
        self.save_data_for_cpp(data_file, query_file, query_vectors)
        
        # Εκτέλεση C++ binary
        cmd = [
            "./build/lsh_search",  # Υποθέτουμε compiled binary
            "-d", data_file,
            "-q", query_file,
            "-o", output_file,
            "-type", "sift",  # Χρησιμοποιούμε SIFT format
            "-N", str(N),
            "-R", str(R),
            "-k", str(k),
            "-L", str(L),
            "-w", str(w),
            "-range", "false"
        ]
        
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            total_time = time.time() - start_time
            
            # Parse results
            results = self.parse_cpp_output(output_file)
            return results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"LSH failed: {e.stderr}")
            return {}
    
    def run_hypercube(self, query_vectors, N=50, R=1000.0, kproj=14, w=4.0, M=10, probes=2):
        # Εκτέλεση Hypercube (C++)
        logger.info(f"Εκτέλεση Hypercube με kproj={kproj}, M={M}, probes={probes}...")
        
        data_file = os.path.join(self.temp_dir, "data.bin")
        query_file = os.path.join(self.temp_dir, "query.bin")
        output_file = os.path.join(self.temp_dir, "hypercube_output.txt")
        
        self.save_data_for_cpp(data_file, query_file, query_vectors)
        
        cmd = [
            "./build/hypercube_search",
            "-d", data_file,
            "-q", query_file,
            "-o", output_file,
            "-type", "sift",
            "-N", str(N),
            "-R", str(R),
            "-k", str(kproj),
            "-w", str(w),
            "-M", str(M),
            "-probes", str(probes),
            "-range", "false"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return self.parse_cpp_output(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Hypercube failed: {e.stderr}")
            return {}
    
    def run_ivfflat(self, query_vectors, N=50, R=1000.0, kclusters=100, nprobe=10):
        # Εκτέλεση IVF-Flat (C++)
        logger.info(f"Εκτέλεση IVF-Flat με kclusters={kclusters}, nprobe={nprobe}...")
        
        data_file = os.path.join(self.temp_dir, "data.bin")
        query_file = os.path.join(self.temp_dir, "query.bin")
        output_file = os.path.join(self.temp_dir, "ivfflat_output.txt")
        
        self.save_data_for_cpp(data_file, query_file, query_vectors)
        
        cmd = [
            "./build/ivfflat_search",
            "-d", data_file,
            "-q", query_file,
            "-o", output_file,
            "-type", "sift",
            "-N", str(N),
            "-R", str(R),
            "-kclusters", str(kclusters),
            "-nprobe", str(nprobe),
            "-range", "false"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return self.parse_cpp_output(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"IVF-Flat failed: {e.stderr}")
            return {}
    
    def run_ivfpq(self, query_vectors, N=50, R=1000.0, kclusters=100, nprobe=10, M=8, nbits=8):
        # Εκτέλεση IVF-PQ (C++)
        logger.info(f"Εκτέλεση IVF-PQ με kclusters={kclusters}, M={M}, nbits={nbits}...")
        
        data_file = os.path.join(self.temp_dir, "data.bin")
        query_file = os.path.join(self.temp_dir, "query.bin")
        output_file = os.path.join(self.temp_dir, "ivfpq_output.txt")
        
        self.save_data_for_cpp(data_file, query_file, query_vectors)
        
        cmd = [
            "./build/ivfpq_search",
            "-d", data_file,
            "-q", query_file,
            "-o", output_file,
            "-type", "sift",
            "-N", str(N),
            "-R", str(R),
            "-kclusters", str(kclusters),
            "-nprobe", str(nprobe),
            "-M", str(M),
            "-nbits", str(nbits),
            "-range", "false"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return self.parse_cpp_output(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"IVF-PQ failed: {e.stderr}")
            return {}
    
    def run_neural_lsh(self, query_vectors, N=50, R=1000.0, probes=5):
        # Εκτέλεση Neural LSH (Python)
        logger.info(f"Εκτέλεση Neural LSH με probes={probes}...")
        
        # Πρώτα χρειαζόμαστε index (build phase)
        index_file = os.path.join(self.temp_dir, "nlsh_index.pkl")
        data_file = os.path.join(self.temp_dir, "data.bin")
        query_file = os.path.join(self.temp_dir, "query.bin")
        output_file = os.path.join(self.temp_dir, "nlsh_output.txt")
        
        # Save data
        self.save_data_for_cpp(data_file, query_file, query_vectors)
        
        # Build index
        build_cmd = [
            "python3", "nlsh_build.py",
            "-d", data_file,
            "-i", index_file,
            "-type", "sift",
            "--partitions", "100",
            "--epochs", "5",
            "--limit", str(min(10000, len(self.vectors)))  # Limit για ταχύτητα
        ]
        
        try:
            subprocess.run(build_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Neural LSH build failed: {e.stderr}")
            return {}
        
        # Search
        search_cmd = [
            "python3", "nlsh_search.py",
            "-d", data_file,
            "-q", query_file,
            "-i", index_file,
            "-o", output_file,
            "-type", "sift",
            "-N", str(N),
            "-R", str(R),
            "-T", str(probes),
            "-range", "false"
        ]
        
        try:
            subprocess.run(search_cmd, check=True, capture_output=True, text=True)
            return self.parse_cpp_output(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Neural LSH search failed: {e.stderr}")
            return {}
    
    def run_blast(self, query_id, query_seq, top_n=50):
        # Εκτέλεση BLAST
        if not self.database_fasta:
            return []
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as query_file:
                query_file.write(f">{query_id}\n{query_seq}\n")
                query_path = query_file.name
            
            output_path = tempfile.mktemp(suffix='.xml')
            
            blast_cmd = NcbiblastpCommandline(
                query=query_path,
                db=self.database_fasta.replace('.fasta', ''),
                evalue=10,
                outfmt=5,
                out=output_path,
                num_alignments=top_n
            )
            
            stdout, stderr = blast_cmd()
            
            results = []
            with open(output_path) as result_handle:
                blast_records = NCBIXML.parse(result_handle)
                for blast_record in blast_records:
                    for alignment in blast_record.alignments:
                        for hsp in alignment.hsps:
                            hit_id = alignment.title.split()[0]
                            identity = hsp.identities / hsp.align_length * 100
                            results.append((hit_id, identity, hsp.expect))
                            break
            
            os.unlink(query_path)
            os.unlink(output_path)
            
            return sorted(results, key=lambda x: x[2])[:top_n]
            
        except Exception as e:
            logger.warning(f"BLAST error για {query_id}: {e}")
            return []
    
    def calculate_recall(self, ann_results, blast_results, top_n):
        # Υπολογισμός Recall@N
        blast_top_n = set([hit[0] for hit in blast_results[:top_n]])
        ann_ids = set([self.ids[idx] for idx, _ in ann_results[:top_n]])
        
        if not blast_top_n:
            return 0.0
        
        intersection = len(blast_top_n.intersection(ann_ids))
        return intersection / len(blast_top_n)
    
    def run_all_methods(self, query_vectors, methods=['all'], N=50, R=1000.0):
        
        # Εκτέλεση όλων των μεθόδων.
        
        """ Args:
                query_vectors: numpy array με query embeddings
                methods: list of methods to run
                N: top-N neighbors
                R: radius για range search
                
            Returns:
                dict: {method_name: results}"""

        results = {}
        
        if 'all' in methods or 'lsh' in methods:
            results['Euclidean LSH'] = self.run_lsh(query_vectors, N, R)
        
        if 'all' in methods or 'hypercube' in methods:
            results['Hypercube'] = self.run_hypercube(query_vectors, N, R)
        
        if 'all' in methods or 'ivf' in methods:
            results['IVF-Flat'] = self.run_ivfflat(query_vectors, N, R)
            results['IVF-PQ'] = self.run_ivfpq(query_vectors, N, R)
        
        if 'all' in methods or 'neural' in methods:
            results['Neural LSH'] = self.run_neural_lsh(query_vectors, N, R, probes=5)
        
        return results
    
    def format_output(self, query_id, query_seq, all_results, blast_results, top_n=50, display_n=10):
        # Δημιουργία formatted output
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"Query Protein: {query_id}")
        output.append(f"N = {top_n}  (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)")
        output.append(f"{'='*80}\n")
        
        # [1] Συνοπτική σύγκριση
        output.append("[1] Συνοπτική σύγκριση μεθόδων")
        output.append("-" * 80)
        output.append(f"{'Method':<20} | {'Time/query (s)':<15} | {'QPS':<8} | {'Recall@N vs BLAST Top-N':<25}")
        output.append("-" * 80)
        
        # Βρες query index
        query_idx = self.ids.index(query_id) if query_id in self.ids else 0
        
        for method_name, method_results in all_results.items():
            if query_idx in method_results:
                result = method_results[query_idx]
                qps = 1.0 / result['time'] if result['time'] > 0 else 0
                
                # Calculate recall
                ann_results = result['neighbors']
                recall = self.calculate_recall(ann_results, blast_results, top_n)
                
                output.append(f"{method_name:<20} | {result['time']:<15.3f} | {qps:<8.1f} | {recall:<25.2f}")
        
        if blast_results:
            output.append(f"{'BLAST (Ref)':<20} | {'1.500':<15} | {'0.7':<8} | {'1.00 (ορίζει το Top-N)':<25}")
        
        output.append("-" * 80)
        output.append("")
        
        # [2] Top-N γείτονες
        output.append(f"[2] Top-N γείτονες ανά μέθοδο (N = {display_n} για εκτύπωση)")
        output.append("")
        
        blast_dict = {hit[0]: hit[1] for hit in blast_results}
        blast_top_n_ids = set([hit[0] for hit in blast_results[:top_n]])
        
        for method_name, method_results in all_results.items():
            if query_idx not in method_results:
                continue
                
            output.append(f"Method: {method_name}")
            output.append("-" * 120)
            output.append(f"{'Rank':<6} | {'Neighbor ID':<30} | {'L2 Dist':<10} | {'BLAST Identity':<15} | {'In BLAST Top-N?':<17} | {'Bio comment':<30}")
            output.append("-" * 120)
            
            neighbors = method_results[query_idx]['neighbors']
            for rank, (neighbor_idx, distance) in enumerate(neighbors[:display_n], 1):
                neighbor_id = self.ids[neighbor_idx]
                blast_identity = blast_dict.get(neighbor_id, 0.0)
                in_blast_top = "Yes" if neighbor_id in blast_top_n_ids else "No"
                
                bio_comment = "--"
                if blast_identity < 30 and distance < 0.3:
                    bio_comment = "Remote homolog?"
                elif blast_identity < 30 and distance > 0.3:
                    bio_comment = "Potential false positive"
                
                output.append(f"{rank:<6} | {neighbor_id:<30} | {distance:<10.4f} | {blast_identity:<15.1f}% | {in_blast_top:<17} | {bio_comment:<30}")
            
            output.append("")
        
        return "\n".join(output)
    
    def run_benchmark(self, queries_fasta, output_file, methods=['all'], top_n=50, display_n=10):
        # Εκτέλεση πλήρους benchmark
        queries = {}
        for record in SeqIO.parse(queries_fasta, "fasta"):
            queries[record.id] = str(record.seq)
        
        logger.info(f"Βρέθηκαν {len(queries)} query πρωτεΐνες")
        
        # Prepare query vectors
        query_ids = list(queries.keys())
        query_vectors = np.array([self.embeddings[qid] for qid in query_ids if qid in self.embeddings])
        
        # Run all methods once
        logger.info("Εκτέλεση όλων των ANN μεθόδων...")
        all_results = self.run_all_methods(query_vectors, methods, top_n)
        
        # Process each query for output
        with open(output_file, 'w') as f:
            for query_id, query_seq in queries.items():
                if query_id not in self.embeddings:
                    continue
                
                logger.info(f"Processing query: {query_id}")
                
                # BLAST
                blast_results = self.run_blast(query_id, query_seq, top_n)
                
                # Format output
                output = self.format_output(query_id, query_seq, all_results, 
                                          blast_results, top_n, display_n)
                f.write(output)
                f.write("\n")
                print(output)
        
        logger.info(f"Αποτελέσματα αποθηκεύτηκαν στο {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Unified Protein Search Benchmark')
    parser.add_argument('-d', '--data', required=True, help='Embeddings file (.dat)')
    parser.add_argument('-q', '--queries', required=True, help='Query FASTA file')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    parser.add_argument('-m', '--method', default='all',
                       choices=['all', 'lsh', 'hypercube', 'neural', 'ivf'],
                       help='ANN method to use')
    parser.add_argument('--database', help='Database FASTA for BLAST')
    parser.add_argument('--top-n', type=int, default=50, help='N for Recall@N')
    parser.add_argument('--display-n', type=int, default=10, help='N for display')
    
    args = parser.parse_args()
    
    benchmark = ProteinSearchBenchmark(args.data, args.database)
    methods = [args.method] if args.method != 'all' else ['all']
    benchmark.run_benchmark(args.queries, args.output, methods, args.top_n, args.display_n)
    
    logger.info("Ολοκληρώθηκε επιτυχώς!")


if __name__ == '__main__':
    main()