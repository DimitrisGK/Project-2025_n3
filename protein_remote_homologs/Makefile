# Makefile για Protein Remote Homolog Detection με Existing Algorithms

.PHONY: help install compile sample-data embeddings blast-db search full-pipeline test clean

# Files
DB_FASTA = swissprot_sample.fasta
QUERY_FASTA = targets_sample.fasta
EMBEDDINGS = protein_vectors.dat
RESULTS = results.txt

# Build directory
BUILD_DIR = build

help:
	@echo "Protein Remote Homolog Detection - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Εγκατάσταση Python dependencies"
	@echo "  make compile        - Compile C++ αλγορίθμους"
	@echo ""
	@echo "Data Preparation:"
	@echo "  make sample-data    - Δημιουργία sample dataset από UniProt"
	@echo "  make embeddings     - Παραγωγή ESM-2 embeddings"
	@echo "  make blast-db       - Δημιουργία BLAST database"
	@echo ""
	@echo "Execution:"
	@echo "  make search         - Εκτέλεση όλων των ANN μεθόδων"
	@echo "  make search-lsh     - Εκτέλεση μόνο LSH"
	@echo "  make search-ivf     - Εκτέλεση μόνο IVF μεθόδων"
	@echo "  make search-neural  - Εκτέλεση μόνο Neural LSH"
	@echo ""
	@echo "Pipeline:"
	@echo "  make full-pipeline  - Εκτέλεση ολόκληρου pipeline"
	@echo "  make test           - Quick test με synthetic data"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Καθαρισμός generated files"
	@echo "  make clean-all      - Καθαρισμός όλων (including build/)"
	@echo ""

install:
	@echo "Εγκατάσταση Python dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed!"

compile:
	@echo "Compiling C++ algorithms..."
	chmod +x compile_algorithms.sh
	./compile_algorithms.sh
	@echo "✓ Compilation completed!"

sample-data:
	@echo "Δημιουργία sample dataset..."
	python create_sample_data.py --mode uniprot \
		--output-db $(DB_FASTA) \
		--output-queries $(QUERY_FASTA)
	@echo "✓ Sample data created!"

embeddings: $(DB_FASTA)
	@echo "Παραγωγή ESM-2 embeddings για $(DB_FASTA)..."
	python protein_embed.py -i $(DB_FASTA) -o $(EMBEDDINGS)
	@echo "✓ Embeddings created: $(EMBEDDINGS)"

blast-db: $(DB_FASTA)
	@echo "Δημιουργία BLAST database..."
	makeblastdb -in $(DB_FASTA) -dbtype prot -out swissprot_sample
	@echo "✓ BLAST database created!"

# Main search with all methods
search: $(EMBEDDINGS) $(QUERY_FASTA)
	@echo "Εκτέλεση unified search με όλες τις μεθόδους..."
	python protein_search_unified.py \
		-d $(EMBEDDINGS) \
		-q $(QUERY_FASTA) \
		-o $(RESULTS) \
		-m all \
		--database $(DB_FASTA) \
		--top-n 50 \
		--display-n 10
	@echo "✓ Search completed! Results in $(RESULTS)"

# Individual method searches
search-lsh: $(EMBEDDINGS) $(QUERY_FASTA)
	@echo "Εκτέλεση LSH μόνο..."
	python protein_search_unified.py \
		-d $(EMBEDDINGS) \
		-q $(QUERY_FASTA) \
		-o results_lsh.txt \
		-m lsh \
		--database $(DB_FASTA)
	@echo "✓ LSH completed!"

search-hypercube: $(EMBEDDINGS) $(QUERY_FASTA)
	@echo "Εκτέλεση Hypercube μόνο..."
	python protein_search_unified.py \
		-d $(EMBEDDINGS) \
		-q $(QUERY_FASTA) \
		-o results_hypercube.txt \
		-m hypercube \
		--database $(DB_FASTA)
	@echo "✓ Hypercube completed!"

search-ivf: $(EMBEDDINGS) $(QUERY_FASTA)
	@echo "Εκτέλεση IVF μεθόδων μόνο..."
	python protein_search_unified.py \
		-d $(EMBEDDINGS) \
		-q $(QUERY_FASTA) \
		-o results_ivf.txt \
		-m ivf \
		--database $(DB_FASTA)
	@echo "✓ IVF methods completed!"

search-neural: $(EMBEDDINGS) $(QUERY_FASTA)
	@echo "Εκτέλεση Neural LSH μόνο..."
	python protein_search_unified.py \
		-d $(EMBEDDINGS) \
		-q $(QUERY_FASTA) \
		-o results_neural.txt \
		-m neural \
		--database $(DB_FASTA)
	@echo "✓ Neural LSH completed!"

# Full pipeline
full-pipeline: install compile
	@echo ""
	@echo "=========================================="
	@echo "ΕΚΤΕΛΕΣΗ ΠΛΗΡΟΥΣ PIPELINE"
	@echo "=========================================="
	@make sample-data
	@make embeddings
	@make blast-db
	@make search
	@echo ""
	@echo "=========================================="
	@echo "✓ PIPELINE ΟΛΟΚΛΗΡΩΘΗΚΕ!"
	@echo "=========================================="
	@echo "Embeddings: $(EMBEDDINGS)"
	@echo "Results: $(RESULTS)"
	@echo "=========================================="

# Quick test with synthetic data
test: compile
	@echo "Quick test με synthetic data..."
	python create_sample_data.py --mode synthetic
	python protein_embed.py -i test_database.fasta -o test_vectors.dat
	python protein_search_unified.py \
		-d test_vectors.dat \
		-q test_queries.fasta \
		-o test_results.txt \
		-m all
	@echo "✓ Test completed! Results in test_results.txt"

# Benchmarks
benchmark-all: $(EMBEDDINGS) $(QUERY_FASTA)
	@echo "Running comprehensive benchmark..."
	@echo "This will take several minutes..."
	@for method in lsh hypercube ivf neural; do \
		echo "Testing $$method..."; \
		time python protein_search_unified.py \
			-d $(EMBEDDINGS) \
			-q $(QUERY_FASTA) \
			-o benchmark_$$method.txt \
			-m $$method \
			--database $(DB_FASTA); \
	done
	@echo "✓ Benchmark completed! Results in benchmark_*.txt"

# Biological analysis
bio-analysis: $(RESULTS)
	@echo "Running biological analysis..."
	python -c "from bio_analysis import BiologicalAnalyzer; \
		analyzer = BiologicalAnalyzer(); \
		print('Biological analysis tools ready')"
	@echo "✓ Use bio_analysis.py for detailed remote homolog analysis"

# Cleanup
clean:
	@echo "Καθαρισμός generated files..."
	rm -f $(EMBEDDINGS)
	rm -f $(RESULTS)
	rm -f results_*.txt
	rm -f benchmark_*.txt
	rm -f test_vectors.dat test_results.txt
	rm -f test_database.fasta test_queries.fasta
	rm -f swissprot_sample.p* swissprot_sample.n* swissprot_sample.h*
	rm -rf __pycache__ modules/__pycache__
	@echo "✓ Cleaned!"

clean-all: clean
	@echo "Καθαρισμός build directory..."
	rm -rf $(BUILD_DIR)
	@echo "✓ All cleaned!"

# Development helpers
check-setup:
	@echo "Checking setup..."
	@which python3 > /dev/null && echo "✓ Python3 found" || echo "✗ Python3 not found"
	@which g++ > /dev/null && echo "✓ g++ found" || echo "✗ g++ not found"
	@which makeblastdb > /dev/null && echo "✓ BLAST+ found" || echo "✗ BLAST+ not found"
	@test -d $(BUILD_DIR) && echo "✓ Build directory exists" || echo "✗ Build directory missing"
	@test -f $(BUILD_DIR)/lsh_search && echo "✓ LSH compiled" || echo "✗ LSH not compiled"
	@test -f $(BUILD_DIR)/hypercube_search && echo "✓ Hypercube compiled" || echo "✗ Hypercube not compiled"
	@test -f $(BUILD_DIR)/ivfflat_search && echo "✓ IVF-Flat compiled" || echo "✗ IVF-Flat not compiled"
	@test -f $(BUILD_DIR)/ivfpq_search && echo "✓ IVF-PQ compiled" || echo "✗ IVF-PQ not compiled"

info:
	@echo "Project Information:"
	@echo "===================="
	@echo "Database: $(DB_FASTA)"
	@echo "Queries: $(QUERY_FASTA)"
	@echo "Embeddings: $(EMBEDDINGS)"
	@echo "Results: $(RESULTS)"
	@echo ""
	@if [ -f $(EMBEDDINGS) ]; then \
		python -c "import pickle; d=pickle.load(open('$(EMBEDDINGS)','rb')); \
			print(f'Embeddings: {len(d[\"embeddings\"])} proteins, dim={d[\"embedding_dim\"]}')"; \
	else \
		echo "Embeddings not generated yet"; \
	fi