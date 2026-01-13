#!/usr/bin/env python3
"""
Δημιουργία sample dataset για testing.
Κατεβάζει μερικές πρωτεΐνες από UniProt για demo.
"""

import requests
from Bio import SeqIO
from io import StringIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_uniprot_sequences(uniprot_ids):
    """
    Κατέβασμα sequences από UniProt.
    
    Args:
        uniprot_ids: List of UniProt IDs
        
    Returns:
        dict: {id: sequence}
    """
    sequences = {}
    
    for uid in uniprot_ids:
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                fasta_io = StringIO(response.text)
                record = next(SeqIO.parse(fasta_io, "fasta"))
                sequences[uid] = str(record.seq)
                logger.info(f"✓ Fetched {uid}: {len(sequences[uid])} aa")
            else:
                logger.warning(f"✗ Failed to fetch {uid}: {response.status_code}")
        
        except Exception as e:
            logger.error(f"✗ Error fetching {uid}: {e}")
    
    return sequences


def create_sample_dataset(output_db='swissprot_sample.fasta', 
                         output_queries='targets_sample.fasta'):
    """
    Δημιουργία sample dataset.
    
    Επιλέγουμε πρωτεΐνες από διαφορετικές οικογένειες για ενδιαφέρουσα ανάλυση.
    """
    logger.info("Δημιουργία sample dataset...")
    
    # Database proteins - διαφορετικές οικογένειες
    database_ids = [
        # Kinases
        'P06493',  # CDK1_HUMAN (Cell division protein kinase 1)
        'P24941',  # CDK2_HUMAN (Cyclin-dependent kinase 2)
        'Q00534',  # CDK6_HUMAN (Cyclin-dependent kinase 6)
        
        # Proteases
        'P00734',  # THRB_HUMAN (Thrombin)
        'P00747',  # PLMN_HUMAN (Plasminogen)
        'P07288',  # PSA_HUMAN (Prostate-specific antigen)
        
        # Dehydrogenases
        'P00325',  # ADH1B_HUMAN (Alcohol dehydrogenase 1B)
        'P07327',  # ADH1A_HUMAN (Alcohol dehydrogenase 1A)
        
        # Heat shock proteins
        'P08107',  # HSP71_HUMAN (Heat shock 70 kDa protein 1)
        'P11021',  # GRP78_HUMAN (Heat shock 70 kDa protein 5)
        
        # Transcription factors
        'P04637',  # P53_HUMAN (Cellular tumor antigen p53)
        'P01308',  # INS_HUMAN (Insulin)
        
        # Immunoglobulins
        'P01857',  # IGHG1_HUMAN (Ig gamma-1 chain C region)
        'P01834',  # IGKC_HUMAN (Ig kappa chain C region)
        
        # Others
        'P69905',  # HBA_HUMAN (Hemoglobin alpha)
        'P68871',  # HBB_HUMAN (Hemoglobin beta)
    ]
    
    # Query proteins - κάποιες από αυτές θα είναι remote homologs
    query_ids = [
        'P06493',  # CDK1 - θα βρει CDK2, CDK6
        'P00325',  # ADH1B - θα βρει ADH1A
        'P08107',  # HSP71 - θα βρει GRP78
        'P69905',  # HBA - θα βρει HBB
    ]
    
    logger.info(f"Fetching {len(database_ids)} database proteins...")
    db_sequences = fetch_uniprot_sequences(database_ids)
    
    logger.info(f"Fetching {len(query_ids)} query proteins...")
    query_sequences = fetch_uniprot_sequences(query_ids)
    
    # Εγγραφή database
    with open(output_db, 'w') as f:
        for uid, seq in db_sequences.items():
            f.write(f">{uid}\n{seq}\n")
    
    logger.info(f"✓ Database αποθηκεύτηκε στο {output_db}")
    
    # Εγγραφή queries
    with open(output_queries, 'w') as f:
        for uid, seq in query_sequences.items():
            f.write(f">{uid}\n{seq}\n")
    
    logger.info(f"✓ Queries αποθηκεύτηκαν στο {output_queries}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SAMPLE DATASET SUMMARY")
    logger.info("="*60)
    logger.info(f"Database proteins: {len(db_sequences)}")
    logger.info(f"Query proteins: {len(query_sequences)}")
    logger.info("\nExpected Remote Homologs:")
    logger.info("  • CDK1 ↔ CDK2, CDK6 (same family, ~40% identity)")
    logger.info("  • ADH1B ↔ ADH1A (isoforms, ~95% identity)")
    logger.info("  • HSP71 ↔ GRP78 (HSP70 family, ~60% identity)")
    logger.info("  • HBA ↔ HBB (globin family, ~42% identity)")
    logger.info("="*60 + "\n")


def create_minimal_test():
    """Δημιουργία minimal test με synthetic sequences."""
    logger.info("Δημιουργία minimal synthetic test...")
    
    # Synthetic sequences για quick testing
    sequences = {
        'PROT_A': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK',
        'PROT_B': 'MKTAYIAKQRQISFVKSHFSRQLEERLGXIEVQAPILSRVGDGTQDNLSGAEK',  # 98% similar
        'PROT_C': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGXXX',  # 95% similar
        'PROT_D': 'MKTAYXXXXXXXXXXXXXXXXXXXXXXGLIEVQAPILSRVGDGTQDNLSGAEK',  # 70% similar
        'PROT_E': 'ARNDCEQGHILKMFPSTWYV' * 3,  # Random
    }
    
    # Save
    with open('test_database.fasta', 'w') as f:
        for pid, seq in sequences.items():
            f.write(f">{pid}\n{seq}\n")
    
    with open('test_queries.fasta', 'w') as f:
        f.write(f">PROT_A\n{sequences['PROT_A']}\n")
    
    logger.info("✓ Test files created: test_database.fasta, test_queries.fasta")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample dataset')
    parser.add_argument('--mode', choices=['uniprot', 'synthetic'], default='uniprot',
                       help='Dataset type')
    parser.add_argument('--output-db', default='swissprot_sample.fasta',
                       help='Output database file')
    parser.add_argument('--output-queries', default='targets_sample.fasta',
                       help='Output queries file')
    
    args = parser.parse_args()
    
    if args.mode == 'uniprot':
        create_sample_dataset(args.output_db, args.output_queries)
    else:
        create_minimal_test()


if __name__ == '__main__':
    main()