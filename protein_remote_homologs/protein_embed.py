#!/usr/bin/env python3
"""
Παραγωγή protein embeddings χρησιμοποιώντας το ESM-2 μοντέλο.
Χρήση: python protein_embed.py -i swissprot.fasta -o protein_vectors.dat
"""

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import pickle
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ESM2Embedder:
    #Κλάση για την παραγωγή embeddings από πρωτεϊνικές ακολουθίες
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", device=None):
        
        #Αρχικοποίηση του ESM-2 μοντέλου.
        
        """Args:
                model_name: Όνομα του ESM-2 μοντέλου
                device: Device για inference (cuda/cpu)"""
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Φόρτωση μοντέλου {model_name} στο {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info("Μοντέλο φορτώθηκε επιτυχώς!")
    
    def embed_sequence(self, sequence, max_length=1024):
        #Δημιουργία embedding για μία πρωτεϊνική ακολουθία.
        
        """ Args:
                sequence: Πρωτεϊνική ακολουθία (string)
                max_length: Μέγιστο μήκος ακολουθίας
            
            Returns:
                numpy array: Embedding vector (mean pooling από το τελευταίο layer)"""
        
        # Περικοπή μεγάλων ακολουθιών
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        # Tokenization
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean pooling από το τελευταίο hidden layer
        # Αγνοούμε τα special tokens (<cls>, <eos>)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # Mask για να αγνοήσουμε padding tokens
        attention_mask = inputs['attention_mask']
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).squeeze()
        
        return embedding.cpu().numpy()
    
    def embed_fasta(self, fasta_file, batch_size=1):
        
        #Δημιουργία embeddings για όλες τις πρωτεΐνες σε ένα FASTA αρχείο.
        
        """ Args:
            fasta_file: Path στο FASTA αρχείο
            batch_size: Batch size για processing
            
            Returns:
                dict: {protein_id: embedding_vector}
                dict: {protein_id: sequence}"""
        
        embeddings = {}
        sequences = {}
        
        logger.info(f"Επεξεργασία FASTA αρχείου: {fasta_file}")
        
        # Διάβασμα FASTA
        records = list(SeqIO.parse(fasta_file, "fasta"))
        logger.info(f"Βρέθηκαν {len(records)} πρωτεΐνες")
        
        # Processing με progress bar
        for record in tqdm(records, desc="Generating embeddings"):
            protein_id = record.id
            sequence = str(record.seq)
            
            try:
                embedding = self.embed_sequence(sequence)
                embeddings[protein_id] = embedding
                sequences[protein_id] = sequence
            except Exception as e:
                logger.warning(f"Σφάλμα στην επεξεργασία {protein_id}: {e}")
                continue
        
        logger.info(f"Δημιουργήθηκαν {len(embeddings)} embeddings επιτυχώς")
        return embeddings, sequences


def save_embeddings(embeddings, sequences, output_file):
    
    #Αποθήκευση embeddings σε αρχείο.
    
    """ Args:
        embeddings: Dict με protein IDs -> embeddings
        sequences: Dict με protein IDs -> sequences
        output_file: Output file path"""

    data = {
        'embeddings': embeddings,
        'sequences': sequences,
        'embedding_dim': list(embeddings.values())[0].shape[0]
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Embeddings αποθηκεύτηκαν στο {output_file}")
    logger.info(f"Διάσταση embeddings: {data['embedding_dim']}")


def main():
    parser = argparse.ArgumentParser(description='Παραγωγή protein embeddings με ESM-2')
    parser.add_argument('-i', '--input', required=True, help='Input FASTA αρχείο')
    parser.add_argument('-o', '--output', required=True, help='Output αρχείο (.dat)')
    parser.add_argument('--model', default='facebook/esm2_t6_8M_UR50D', 
                       help='ESM-2 model name')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Δημιουργία embedder
    embedder = ESM2Embedder(model_name=args.model, device=args.device)
    
    # Παραγωγή embeddings
    embeddings, sequences = embedder.embed_fasta(args.input)
    
    # Αποθήκευση
    save_embeddings(embeddings, sequences, args.output)
    
    logger.info("Ολοκληρώθηκε επιτυχώς!")


if __name__ == '__main__':
    main()