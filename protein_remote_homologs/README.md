# Εγκατάσταση
pip install -r requirements.txt

# Compile όλους τους αλγορίθμους
chmod +x compile_algorithms.sh
./compile_algorithms.sh

# Ή με Makefile
make compile

## Χρήση - Full Pipeline

### Βήμα 1: Δημιουργία Sample Data (για testing)

```bash
python create_sample_data.py --mode uniprot \
  --output-db swissprot_sample.fasta \
  --output-queries targets_sample.fasta
```

### Βήμα 2: Παραγωγή Embeddings

```bash
python3 protein_embed.py \
  -i swissprot_sample.fasta \
  -o protein_vectors.dat \
  --device cuda  # ή cpu
```

**Output:** `protein_vectors.dat` με ESM-2 embeddings

### Βήμα 3: Προετοιμασία BLAST Database

```bash
makeblastdb -in swissprot_sample.fasta -dbtype prot -out swissprot_sample
```

### Βήμα 4: Εκτέλεση Unified Search

```bash
python3 protein_search_unified.py \
  -d protein_vectors.dat \
  -q targets_sample.fasta \
  -o results.txt \
  -m all \
  --database swissprot_sample.fasta \
  --top-n 50 \
  --display-n 10
```

**Παράμετροι:**
- `-d, --data`: Embeddings file (.dat)
- `-q, --queries`: Query FASTA
- `-o, --output`: Output file
- `-m, --method`: Μέθοδος (all, lsh, hypercube, ivf, neural)
- `--database`: FASTA για BLAST
- `--top-n`: N για Recall@N (default: 50)
- `--display-n`: N για εμφάνιση (default: 10)

## Υλοποιημένες Μέθοδοι

### 1. **Euclidean LSH** (C++)
```cpp
// Από την 1η εργασία
run_lsh(data, queries, N, R, k=4, L=5, w=4.0, seed, output_file, do_range);
```
- **k**: Hash functions ανά table
- **L**: Αριθμός hash tables
- **w**: Bucket width

### 2. **Hypercube Projection** (C++)
```cpp
// Από την 1η εργασία
run_hypercube(data, queries, N, R, kproj=14, w=4.0, M=10, probes=2, seed, output_file, do_range);
```
- **kproj**: Διαστάσεις projection
- **M**: Max σημεία για έλεγχο
- **probes**: Max κορυφές για έλεγχο

### 3. **IVF-Flat** (C++)
```cpp
// Από την 2η εργασία
run_ivfflat(data, queries, N, R, kclusters=100, nprobe=10, seed, output_file, do_range);
```
- **kclusters**: Αριθμός clusters
- **nprobe**: Clusters για έλεγχο

### 4. **IVF-PQ** (C++)
```cpp
// Από την 2η εργασία
run_ivfpq(data, queries, N, R, kclusters=100, nprobe=10, M=8, nbits=8, seed, output_file, do_range);
```
- **M**: Subquantizers
- **nbits**: Bits ανά subquantizer

### 5. **Neural LSH** (Python)
```bash
# Build phase
python nlsh_build.py -d data.bin -i index.pkl -type sift --partitions 100

# Search phase
python nlsh_search.py -d data.bin -q query.bin -i index.pkl -o output.txt -T 5
```
- **partitions**: Αριθμός partitions (KaHIP)
- **T (probes)**: Partitions για έλεγχο

## Μορφή Εξόδου

Για κάθε query protein:

```
================================================================================
Query Protein: P06493
N = 50  (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)
================================================================================

[1] Συνοπτική σύγκριση μεθόδων
--------------------------------------------------------------------------------
Method               | Time/query (s)  | QPS      | Recall@N vs BLAST Top-N
--------------------------------------------------------------------------------
Euclidean LSH        | 0.020           | 50       | 0.92
Hypercube            | 0.030           | 33       | 0.88
Neural LSH           | 0.010           | 100      | 0.95
IVF-Flat             | 0.008           | 125      | 0.93
IVF-PQ               | 0.005           | 200      | 0.90
BLAST (Ref)          | 1.500           | 0.7      | 1.00 (ορίζει το Top-N)
--------------------------------------------------------------------------------

[2] Top-N γείτονες ανά μέθοδο (N = 10 για εκτύπωση)

Method: Euclidean LSH
------------------------------------------------------------------------------------------------------------------------
Rank   | Neighbor ID                    | L2 Dist    | BLAST Identity  | In BLAST Top-N?   | Bio comment
------------------------------------------------------------------------------------------------------------------------
1      | P24941                         | 0.1500     | 42.5%           | Yes               | Remote homolog?
2      | Q00534                         | 0.1800     | 38.2%           | Yes               | Remote homolog?
...
```

## Βιολογική Αξιολόγηση

Για λεπτομερή βιολογική ανάλυση των remote homologs:

```python
from bio_analysis import BiologicalAnalyzer

analyzer = BiologicalAnalyzer()

# Identify candidates
candidates = analyzer.identify_remote_homologs(
    query_id='P06493',
    neighbors=[('P24941', 0.15), ('Q00534', 0.18)],
    blast_results=[('P24941', 42.5, 1e-50)],
    l2_threshold=0.3,
    identity_threshold=30
)

# Generate report
report = analyzer.format_remote_homolog_report('P06493', candidates)
print(report)
```

**Output:**
```
================================================================================
REMOTE HOMOLOG ANALYSIS για Query: P06493
================================================================================

Candidate #1: P24941
--------------------------------------------------------------------------------
L2 Distance: 0.1500
BLAST Identity: 42.5% (Twilight Zone)
Similarity Score: 0.752

Biological Evidence:
  • Common Pfam: PF00069 (Protein kinase domain)
  • GO: 15 terms
  • Common EC: 2.7.11.22

Detailed Comparison:
  Query: Cyclin-dependent kinase 1 (Homo sapiens)
  Neighbor: Cyclin-dependent kinase 2 (Homo sapiens)
  Pfam Jaccard: 0.850
  GO Jaccard: 0.680
```

## Αυτοματοποίηση με Makefile

```bash
# Full pipeline
make full-pipeline

# Individual steps
make install           # Install dependencies
make sample-data       # Create sample dataset
make embeddings        # Generate embeddings
make blast-db          # Build BLAST database
make search            # Run search

# Testing
make test              # Quick test με synthetic data

# Cleanup
make clean             # Remove generated files
```

## Παραδείγματα Χρήσης

### Example 1: Μόνο LSH

```bash
python protein_search_unified.py \
  -d protein_vectors.dat \
  -q targets.fasta \
  -o results_lsh.txt \
  -m lsh
```

### Example 2: IVF μέθοδοι

```bash
python protein_search_unified.py \
  -d protein_vectors.dat \
  -q targets.fasta \
  -o results_ivf.txt \
  -m ivf
```

### Example 3: Πλήρης ανάλυση με BLAST

```bash
python protein_search_unified.py \
  -d protein_vectors.dat \
  -q targets.fasta \
  -o results_full.txt \
  -m all \
  --database swissprot.fasta \
  --top-n 100
```

## Προσαρμογή Παραμέτρων

### Για μεγαλύτερη ακρίβεια:
Επεξεργασία του `protein_search_unified.py`:

```python
# LSH
results['Euclidean LSH'] = self.run_lsh(query_vectors, N, R, k=6, L=30, w=3.0)

# Hypercube
results['Hypercube'] = self.run_hypercube(query_vectors, N, R, kproj=16, M=20, probes=3)

# IVF
results['IVF-Flat'] = self.run_ivfflat(query_vectors, N, R, kclusters=200, nprobe=20)
```

### Για μεγαλύτερη ταχύτητα:
```python
# LSH με λιγότερα tables
results['Euclidean LSH'] = self.run_lsh(query_vectors, N, R, k=3, L=3)

# IVF-PQ αντί IVF-Flat
results['IVF-PQ'] = self.run_ivfpq(query_vectors, N, R, M=16, nbits=6)

# Neural LSH με λιγότερα probes
results['Neural LSH'] = self.run_neural_lsh(query_vectors, N, R, probes=3)
```

## Performance Tips

### 1. **GPU Acceleration για ESM-2**
```bash
python protein_embed.py -i data.fasta -o vectors.dat --device cuda
```

### 2. **Parallel Processing**
Για πολλά queries, split σε batches:
```bash
# Split queries
split -l 10 targets.fasta targets_batch_

# Run in parallel
for batch in targets_batch_*; do
    python protein_search_unified.py -d vectors.dat -q $batch -o results_$batch.txt &
done
wait
```

### 3. **Memory Optimization**
Για μεγάλες βάσεις (>100K πρωτεΐνες):
```python
# Limit Neural LSH training data
--limit 10000  # στο nlsh_build.py
```

## Troubleshooting

### "C++ binary not found"
```bash
# Rebuild
./compile_algorithms.sh

# Check binaries
ls -lh build/
```

### "BLAST not found"
```bash
# Install
sudo apt-get install ncbi-blast+

# Or add to PATH
export PATH=$PATH:/path/to/blast/bin
```

### "Out of memory" (ESM-2)
```bash
# Use smaller model
python protein_embed.py -i data.fasta -o vectors.dat \
  --model facebook/esm2_t6_8M_UR50D  # Smallest model

# Or process in batches
```

### "Dimension mismatch"
Όλα τα vectors πρέπει να έχουν την ίδια διάσταση. Τα ESM-2 embeddings είναι:
- `esm2_t6_8M_UR50D`: 320 dimensions
- `esm2_t12_35M_UR50D`: 480 dimensions
- `esm2_t30_150M_UR50D`: 640 dimensions

## Αναμενόμενα Αποτελέσματα

### Remote Homolog Examples

**Kinases (CDK family):**
- Query: CDK1 (P06493)
- Neighbors: CDK2 (P24941), CDK6 (Q00534)
- BLAST Identity: ~40% (Twilight Zone)
- Evidence: Κοινό Pfam domain (PF00069), παρόμοια GO terms

**Globins:**
- Query: Hemoglobin α (P69905)
- Neighbor: Hemoglobin β (P68871)
- BLAST Identity: ~42%
- Evidence: Κοινό Pfam domain (PF00042), oxygen transport function

### Performance Benchmarks

Σε dataset 10,000 πρωτεϊνών (SIFT-like dimensions ~320):

| Method | QPS | Recall@50 | Build Time |
|--------|-----|-----------|------------|
| LSH | 50-100 | 0.85-0.92 | ~5s |
| Hypercube | 30-50 | 0.80-0.88 | ~3s |
| IVF-Flat | 100-150 | 0.90-0.95 | ~30s |
| IVF-PQ | 150-250 | 0.85-0.92 | ~60s |
| Neural LSH | 80-120 | 0.92-0.97 | ~5min |
| BLAST | 0.5-1 | 1.00 | N/A |

## Citation

```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and others},
  journal={Science},
  volume={379},
  pages={1123--1130},
  year={2023}
}
```