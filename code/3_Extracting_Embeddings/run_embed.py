"""
Encodes biological sequences from a FASTA file into numerical embeddings.

Each resulting embedding, along with its corresponding label, is saved as a
separate binary '.pickle' file.

Inputs:
- A FASTA file containing sequences to be encoded.
- The desired embedding method ('ohe', 'esm2', etc.).

Outputs:
- A directory containing one '.pickle' file per sequence. Each file contains
  the numerical embedding and its binary label.
"""

import os
import sys
import argparse
import _pickle as cPickle

import numpy as np
from tqdm import tqdm

sys.path.append("utils/")
import encoders as en

def read_fasta(path: str):
    """
    Reads a FASTA file and returns a dictionary of sequences.

    Args:
        path (str): The file path to the FASTA file.

    Returns:
        Dict[str, str]: A dictionary where keys are the sequence IDs 
                        and values are the corresponding sequences.
    """
    with open(path, 'r') as file:
        content = file.read()
    
    sequences = {}
    for i in content.split(">")[1:]:
        sequence_id = i.split("\n")[0]
        sequence = "".join(i.split("\n")[1:])
        sequences[sequence_id] = sequence
    return sequences

def write_pickle(path: str, sequence_id: str, x, y):
    """
    Serializes and saves a sequence's embedding (x) and label (y) to a pickle file.

    Two objects (the embedding array and the label array) are dumped sequentially
    into a single binary file.

    Args:
        path (str): The output directory where the file will be saved.
        sequence_id (str): The ID of the sequence, used as the filename.
        x (np.ndarray): The numerical embedding of the sequence.
        y (np.ndarray): The numerical label of the sequence.
    """
    # 'wb' mode is crucial for writing binary data with pickle.
    with open(os.path.join(path, sequence_id + ".pickle"), 'wb') as file:
        cPickle.dump(x, file)  # Save the embedding array first.
        cPickle.dump(y, file)  # Then save the label array.

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script with default values for arguments.")
    # Define arguments and their default values
    parser.add_argument("--embed", type=str, default='ohe', help="One of {'ohe', 'bl62', 'esm2', 'prott5', 'xtrimopglm_10b', 'xtrimopglm_100b'}.")
    parser.add_argument("--fasta_path", type=str, default='../Training_Applm_RF_Model/1_train_splits_fa/ex1/train.fa', help='e.g. ../Training_Applm_RF_Model/1_train_splits_fa/ex1/train.fa')
    parser.add_argument("--output_directory", type=str, default='../Training_Applm_RF_Model/0_embeddings/avgpool_ohe', help="e.g. ../Training_Applm_RF_Model/0_embeddings/avgpool_ohe")

    args = parser.parse_args()
    embed = args.embed
    fasta_path = args.fasta_path
    output_directory = args.output_directory

    if embed == 'ohe':
        encoder = en.OHE_Encoder()
    elif embed == 'bl62':
        encoder = en.BL62_Encoder()
    elif embed == 'esm2':
        encoder = en.ESM2_Encoder()
    elif embed == 'prott5':
        encoder = en.T5_Encoder()
    elif embed == 'xtrimopglm_10b':
        encoder = en.xTrimoPGLM_Encoder("biomap-research/proteinglm-10b-mlm")
    elif embed == 'xtrimopglm_100b':
        encoder = en.xTrimoPGLM_Encoder("biomap-research/proteinglm-100b-int4")
    else:
        print(f"Error: Unrecognized embedding option '{embed}'. Exiting.")
        sys.exit(1) # Exit with an error code.

    output_directory = os.path.join(output_directory, embed)
    os.makedirs(output_directory, exist_ok=True)

    # Read the fasta file
    # *** The name for each sequence is determined by the sequence_id / header in the fasta ***
    sequences = read_fasta(fasta_path)

    desc = f'Encoding sequences with {embed}'
    for sequence_id, sequence in tqdm(sequences.items(), desc=desc):
        x = encoder.encode_sequence(sequence)

        ### Define the sequence label
        # In our example, negative examples are explicitly marked with '_NEG'.
        label = np.array([0.]) if '_NEG' in sequence_id else np.array([0.])

        ### Save the embedding as an instance
        write_pickle(output_directory, sequence_id, x, label)

    print(f"\nProcessing complete. {len(sequences)} embeddings saved.")
    
