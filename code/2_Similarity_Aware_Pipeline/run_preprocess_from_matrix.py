import _pickle as cPickle
import numpy as np
import h5py
from operator import itemgetter
import copy
import random
from tqdm import tqdm
from collections import Counter, defaultdict
import argparse
import os

def get_seq_keys(path):
    with open(path, "r") as file:
        s1 = file.read()
    seq_keys = [i.split("\n")[0] for i in s1.split(">")[1:]]
    return sorted(seq_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with default values for arguments.")
    # Define arguments and their default values
    parser.add_argument("--fasta_path", type=str, default='example_input/ex3.fa')
    parser.add_argument("--matrix_path", type=str, default='example_input/ex3_pairwise_ident.npy')
    parser.add_argument("--output_directory", type=str, default='example_intermediate')
    parser.add_argument("--coverage_control", type=float, default=0.25)

    args = parser.parse_args()
    fasta_path = args.fasta_path
    matrix_path = args.matrix_path
    output_directory = args.output_directory
    coverage_control = args.coverage_control
    project_name = fasta_path.split("/")[-1].split(".")[0]

    ### Step 1: The sequence to index mapping object
    seq_keys = get_seq_keys(fasta_path)
    key_to_idx = defaultdict(int)
    key_to_idx['placeholder'] = 0
    for ni, i in enumerate(seq_keys):
        key_to_idx[i] = ni+1

    with open(os.path.join(output_directory, "{:}_key_to_idx.pickle".format(project_name)), 'wb') as file:
        cPickle.dump(key_to_idx, file)

    ### Step 3: Create the h5 object which stores and retrieves the pairwise identities

    # Create an HDF5 file
    # The actual H5 data structure will reserve the first row and first column as a 
    # place-holder for any non-existing keys, which it will then return 0. i.e. the
    # H5 structure will be of size `(len(seq_keys)+1) x (len(seq_keys)+1)`

    loaded_array = np.load(matrix_path)

    with h5py.File(os.path.join(output_directory, '{:}.h5'.format(project_name)), 'w') as hdf:
        hdf.create_dataset('0', data=np.zeros(loaded_array.shape[0] + 1, dtype=float))  # index 0 is reserved for placeholder
        for ni, i in enumerate(tqdm(seq_keys)):
            ident_list = loaded_array[ni]   # identities of sequence i against all of seq_keys
            ident_list = np.concatenate([np.array([0.]), ident_list])   # index 0 is reserved for placeholder
            hdf.create_dataset(str(key_to_idx[i]), data=ident_list)

    print("{:} written.".format(os.path.join(output_directory, '{:}.h5'.format(project_name))))