"""
Processes sequence similarity search results to create a pairwise identity matrix
and stores it in an efficient HDF5 database.

Input:
- A FASTA file (`--fasta_path`) containing the sequences.
- A formatted ssearch output file (`--ssearch_output_path`) with columns:
  query_id, subject_id, identity, query_len, subject_len, alignment_len.

Output:
- A pickled dictionary mapping sequence keys to indices.
- A NumPy array (`.npy`) of the pairwise identity matrix.
- An HDF5 file (`.h5`) for efficient data retrieval.
"""

import _pickle as cPickle
from operator import itemgetter
import copy
import random
from collections import Counter, defaultdict
import argparse
import os

import numpy as np
import h5py
from tqdm import tqdm


def get_seq_keys(path):
    # Reads a FASTA file and extracts the sequence identifiers from the header lines.
    with open(path, "r") as file:
        s1 = file.read()
    seq_keys = [i.split("\n")[0] for i in s1.split(">")[1:]]
    return sorted(seq_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ssearch output to create a pairwise identity matrix and HDF5 database.")
    # Define arguments and their default values
    parser.add_argument("--fasta_path", type=str, default='example_input/ex3.fa')
    parser.add_argument("--ssearch_output_path", type=str, default='../1_Computing_Similarity/example_output/ex3_formatted')
    parser.add_argument("--output_directory", type=str, default='example_intermediate')
    parser.add_argument("--coverage_control", type=float, default=0.25)

    args = parser.parse_args()
    fasta_path = args.fasta_path
    ssearch_output_path = args.ssearch_output_path
    output_directory = args.output_directory
    coverage_control = args.coverage_control

    # Extract a project name from the input FASTA file path for naming output files.
    project_name = fasta_path.split("/")[-1].split(".")[0]
    # Ensure the output directory exists.
    os.makedirs(output_directory, exist_ok=True)

    ### Step 1: The sequence to index mapping object
    seq_keys = get_seq_keys(fasta_path)
    key_to_idx = defaultdict(int)
    key_to_idx['placeholder'] = 0
    for ni, i in enumerate(seq_keys):
        key_to_idx[i] = ni+1

    with open(os.path.join(output_directory, "{:}_key_to_idx.pickle".format(project_name)), 'wb') as file:
        cPickle.dump(key_to_idx, file)
    print(f"Key-to-index map saved to {output_path}")

    ### Step 2: Format identity matrix from ssearch36 formatted output
    with open(ssearch_output_path, 'r') as file:
        content = file.read().strip().split("\n")

    # Initialize a square matrix of zeros
    ident_mat = np.zeros((len(seq_keys), len(seq_keys)))

    for i in tqdm(content):
        if len(i) > 0:
            row = i.split("\t")
            id1 = row[0]
            id2 = row[1]
            identity = float(row[2])
            min_len = np.minimum(int(row[3]), int(row[4]))
            aln_len = int(row[5])

            # custom coverage control
            coverage = min_len/aln_len
            if coverage > coverage_control:
                ident_mat[seq_keys.index(id1), seq_keys.index(id2)] = identity
            # Else remain 0

    np.save(os.path.join(output_directory, '{:}_pairwise_ident.npy'.format(project_name)), ident_mat)
    print(f"Pairwise identity matrix saved to {output_path}")

    ### Step 3: Create the h5 object which stores and retrieves the pairwise identities

    # Create an HDF5 file
    # The actual H5 data structure will reserve the first row and first column as a 
    # place-holder for any non-existing keys, which it will then return 0. i.e. the
    # H5 structure will be of size `(len(seq_keys)+1) x (len(seq_keys)+1)`
    with h5py.File(os.path.join(output_directory, '{:}.h5'.format(project_name)), 'w') as hdf:
        hdf.create_dataset('0', data=np.zeros(ident_mat.shape[0] + 1, dtype=float))  # index 0 is reserved for placeholder
        for ni, i in enumerate(tqdm(seq_keys)):
            ident_list = ident_mat[ni]   # identities of sequence i against all of seq_keys
            ident_list = np.concatenate([np.array([0.]), ident_list])   # index 0 is reserved for placeholder
            hdf.create_dataset(str(key_to_idx[i]), data=ident_list)

    print(f"\n{h5_path} written successfully.")
    print("Script finished.")