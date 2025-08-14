"""
Performs similarity-aware partitioning of a sequence dataset for cross-validation.

Inputs:
- A FASTA file containing all sequences.
- A pre-computed key-to-index mapping (pickle file).
- A pre-computed pairwise identity database (HDF5 file).
- Similarity thresholds (ts, tc) to control partitioning.

Outputs:
- A set of directories, one for each partitioning strategy.
- Within each directory, a series of FASTA files (`split_0.fa`, `split_1.fa`, ...),
  each containing the sequences for one cross-validation fold.
"""

import os
import sys
import argparse
import _pickle as cPickle
from collections import Counter, defaultdict
import copy
import random
from operator import itemgetter

import numpy as np
import h5py
from tqdm import tqdm

sys.path.append("utils/")
import sequenceDatabaseObject as sdo
import utils as su


def read_fasta(filepath):
    """
    Reads a FASTA file and returns a dictionary of sequences.

    The function parses a file in FASTA format, where each sequence is
    preceded by a header line starting with '>'. It extracts the sequence
    identifier from the header and the corresponding sequence.

    Args:
        filepath: The path to the FASTA file.

    Returns:
        A dictionary where keys are sequence identifiers (str) and
        values are the corresponding protein or nucleotide sequences (str).
    """
    with open(filepath, "r") as file:
        s1 = file.read()
    s1 = s1.split(">")[1:]
    fasta_dict = {}
    for i in s1:
        seq_id = i.split("\n")[0]
        seq = ''.join(i.split("\n")[1:])
        fasta_dict[seq_id] = seq
    return fasta_dict

def write_splits_to_fa(pos_splits, neg_splits, sequences, output_path = 'example_output'):
    """
    Writes the partitioned positive and negative sequence splits to FASTA files.

    For each split (fold), it combines the positive and negative sequence IDs,
    retrieves their sequences, and writes them to a single FASTA file named
    'split_{k}.fa'.

    Args:
        pos_splits (dict[int, list[str]]): A dictionary mapping a split index (key)
            to a list of positive sequence IDs (value).
        neg_splits (dict[int, list[str]]): A dictionary mapping a split index to a
            list of negative sequence IDs.
        sequences (dict[str, str]): A dictionary mapping sequence IDs to their
            actual sequences.
        output_path (str): The directory where the output FASTA files will be saved.
    """
    split_keys = sorted(list(pos_splits.keys()))
    for nk, k in enumerate(split_keys):
        fasta_content = ''
        for seqid in pos_splits[k]:
            fasta_content += ">{:}\n{:}\n".format(seqid, sequences[seqid])
        for seqid in neg_splits[k]:
            fasta_content += ">{:}\n{:}\n".format(seqid, sequences[seqid])
        with open(os.path.join(output_path, 'split_{:}.fa'.format(nk)), 'w') as file:
            file.write(fasta_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with default values for arguments.")
    # Define arguments and their default values
    parser.add_argument("--fasta_path", type=str, default='example_input/ex3.fa')
    parser.add_argument("--keys_to_idx_path", type=str, default='example_intermediate/ex3_key_to_idx.pickle')
    parser.add_argument("--hdf5_path", type=str, default='example_intermediate/ex3.h5')
    parser.add_argument("--output_directory", type=str, default='example_output/partitioned_fasta')
    parser.add_argument("--ts", type=float, default=0.4)
    parser.add_argument("--tc", type=float, default=0.5)

    args = parser.parse_args()
    fasta_path = args.fasta_path
    keys_to_idx_path = args.keys_to_idx_path
    hdf5_path = args.hdf5_path
    output_directory = args.output_directory
    ts = args.ts
    tc = args.tc

    # Step 1. Set-up sequence database object
    print("Step 1: Initializing sequence database...")
    seqdb = sdo.SeqDB(fasta_path, keys_to_idx_path, hdf5_path)
    pos_keys = seqdb.get_positive_keys()
    neg_keys = seqdb.get_negative_keys()
    
    sequences = read_fasta(fasta_path)
    print(f"Loaded {len(pos_keys)} positive keys and {len(neg_keys)} negative keys.")

    ### Step 2. Split positives
    print("\nSteps 2 & 3: Splitting positives and removing violations...")
    pos_splits = su.get_positive_splits(pos_keys, seqdb)

    ### Step 3. Remove violations from positives
    pos_splits_1 = su.remove_violations(pos_splits, seqdb, ts)

    ### Step 4. Split negatives without prior negative splits
    print("\nStep 4: Assigning negatives to splits (No Balance strategy)...")
    neg_splits_1 = su.get_negative_splits_seeded(pos_splits_1, 
        dict(), # No prior negative splits
        seqdb, 
        ts, 
        neg_keys, 
        tc, 
        check_violations_at_the_end=True    # If no prior negative splits, check violations at the end
        )

    os.makedirs(os.path.join(output_directory, 'No_Balance'), exist_ok=True)
    write_splits_to_fa(pos_splits_1, neg_splits_1, sequences, os.path.join(output_directory, 'No_Balance'))

    ### Step 5. Hard Balance
    print("\nStep 5: Applying Hard Balance strategy...")
    neg_splits_balanced = su.get_balance(pos_splits_1, neg_splits_1)

    os.makedirs(os.path.join(output_directory, 'Hard_Balance'), exist_ok=True)
    write_splits_to_fa(pos_splits_1, neg_splits_balanced, sequences, os.path.join(output_directory, 'Hard_Balance'))

    ### Step 6. Length Control
    print("\nStep 6: Applying Length Control strategy...")
    neg_splits_lenth_control = su.get_length_control(pos_splits_1, neg_splits_1, seqdb)

    os.makedirs(os.path.join(output_directory, 'Length_Control'), exist_ok=True)
    write_splits_to_fa(pos_splits_1, neg_splits_lenth_control, sequences, os.path.join(output_directory, 'Length_Control'))

    ### Step 7. Minimal
    print("\nStep 7: Applying Minimal strategy...")
    pos_splits_minimal, neg_splits_minimal = su.get_minimal(pos_splits_1, neg_splits_1, seqdb, min_pos=50, min_neg=50, tc=tc)

    os.makedirs(os.path.join(output_directory, 'Minimal'), exist_ok=True)
    write_splits_to_fa(pos_splits_minimal, neg_splits_minimal, sequences, os.path.join(output_directory, 'Minimal'))
    
    print("\nScript finished. All partitioned FASTA files have been generated.")

# Example command to run this script from the terminal.
# python run_similarity_aware_pipeline.py \
#   --fasta_path ../1_Computing_Similarity/example_input/ex3.fa \
#   --keys_to_idx_path example_intermediate/ex3_key_to_idx.pickle \
#   --hdf5_path example_intermediate/ex3.h5 \
#   --output_directory example_output/partitioned_fasta \
#   --ts 0.4 \
#   --tc 0.5