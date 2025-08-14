import _pickle as cPickle
import numpy as np
import h5py
from operator import itemgetter
import copy
import random
from tqdm import tqdm
from collections import Counter, defaultdict

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

def read_pickle(path):
    """
    Reads a binary file created by pickle.

    Args:
        path: The path to the pickle file.

    Returns:
        The Python object deserialized from the pickle file. The exact type
        depends on what was originally saved.
    """
    with open(path, "rb") as file:
        obj = cPickle.load(file)
    return obj

class SeqDB:
    """
    A database class to manage sequence data, identity matrices, and metadata.

    This class loads sequence data from a FASTA file, a mapping of sequence keys
    to indices from a pickle file, and a sequence identity matrix from an HDF5 file.
    It provides methods to access sequences and identity scores.

    Args:
        fasta_path: The path the fasta file.
        keys_to_idx_path: The path to the keys_to_idx pickle file.
        hdf5_path: The path to the h5 file.

    Attributes:
        fasta_path (str): Path to the input FASTA file.
        keys_to_idx_path (str): Path to the pickle file mapping keys to indices.
        hdf5_path (str): Path to the HDF5 file containing the identity matrix.
        seqs (Dict[str, str]): A dictionary of sequences loaded from the FASTA file.
        pos_seq_keys (List[str]): A list of sequence keys identified as positive examples.
        neg_seq_keys (List[str]): A list of sequence keys identified as negative examples.
        keys_to_idx (Dict[str, int]): A mapping from sequence keys to HDF5 indices.
        keys (List[str]): A list of all sequence keys.
        file (h5py.File): An open HDF5 file handle.
    """
    def __init__(self, fasta_path, keys_to_idx_path, hdf5_path):
        """Initializes the SeqDB object by loading data from specified paths."""
        self.fasta_path = fasta_path
        self.keys_to_idx_path = keys_to_idx_path
        self.hdf5_path = hdf5_path

        self.seqs = read_fasta(self.fasta_path)
        self.pos_seq_keys = []
        self.neg_seq_keys = []
        for k in self.seqs.keys():
            if "_NEG" in k:
                self.neg_seq_keys.append(k)
            else:
                self.pos_seq_keys.append(k)
        self.keys_to_idx = read_pickle(self.keys_to_idx_path)
        self.keys = list(self.keys_to_idx.keys())[1:]
        self.file = h5py.File(self.hdf5_path, 'r')

    def get_seq(self, seq_id):
        """
        Retrieves one or more sequences by their identifier(s).

        Args:
            seq_id: A single sequence ID (str) or a list/tuple of sequence IDs.

        Returns:
            The corresponding sequence (str), a list of sequences, or None if the
            input type is invalid.
        """
        if isinstance(seq_id, str):
            return self.seqs[seq_id]
        elif isinstance(seq_id, (list, tuple)):
            return [self.seqs[seq_id] for s in seq_id]
        else:
            return None
    
    def _get_ident(self, id1, id2):
        """
        Internal helper to retrieve the identity score between two sequences.

        Args:
            id1: The first sequence identifier.
            id2: The second sequence identifier.

        Returns:
            The identity score from the HDF5 file.
        """
        idx1, idx2 = self.keys_to_idx[id1], self.keys_to_idx[id2]
        return self.file[str(idx1)][idx2]

    def get_ident(self, id1, id2):
        """
        Retrieves the identity score(s) between sequences.

        This method handles two cases:
        1. If both id1 and id2 are single strings, it returns a single identity score.
        2. If both are lists/tuples, it returns a NumPy matrix of identity scores.
           The matrix dimensions will correspond to the lengths of id1 and id2.

        Args:
            id1: A single sequence ID (str) or a list/tuple of IDs.
            id2: A single sequence ID (str) or a list/tuple of IDs.

        Returns:
            A single identity score or a NumPy array of scores.
        """
        if isinstance(id1, (list, tuple)) and isinstance(id2, (list, tuple)):
            rotate = False
            if len(id2) > len(id1):
                rotate = True
                id1, id2 = id2, id1
            hdf5_idx = self.get_hdf5_mapped_idx(id1)
            ident_mat = []
            for s in id2:
                ident_mat.append(self.get_ident_list(s)[hdf5_idx])
            ident_mat = np.array(ident_mat)
            if not rotate:
                ident_mat = ident_mat.T
            return np.array(ident_mat)

        idx1, idx2 = self.keys_to_idx[id1], self.keys_to_idx[id2]
        return self.file[str(idx1)][idx2]

    def get_ident_list(self, id1):
        """
        Retrieves the entire row of identity scores for a given sequence ID.

        Args:
            id1: The sequence identifier for which to retrieve identity scores.

        Returns:
            A NumPy array containing the identity scores of id1 against all other
            sequences in the HDF5 file.
        """
        idx1 = self.keys_to_idx[id1]
        return np.array(self.file[str(idx1)])

    def get_negative_keys(self):
        """Returns the list of keys identified as negative examples."""
        return self.neg_seq_keys

    def get_positive_keys(self):
        """Returns the list of keys identified as positive examples."""
        return self.pos_seq_keys

    def get_hdf5_mapped_idx(self, keys):
        """
        Converts sequence key(s) to their corresponding HDF5 index/indices.

        Args:
            keys: A single sequence key (str) or a list/tuple of keys.

        Returns:
            A single integer index, a NumPy array of indices, or None if the
            input type is invalid.
        """
        if isinstance(keys, str):
            return self.keys_to_idx[keys]
        elif isinstance(keys, (list, tuple)):
            idx_list = []
            for k in keys:
                idx_list.append(self.keys_to_idx[k])
            return np.array(idx_list)
        else:
            return None
        