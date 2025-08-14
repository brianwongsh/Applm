"""
Trains a Random Forest classifier on pre-computed sequence embeddings, evaluates
it on a test set, and saves the predictions.

Inputs:
- A directory containing 'train.fa' and 'test.fa' files.
- A directory containing the corresponding pre-computed embeddings in .pickle format.
- The embedding type used (e.g., 'ohe', 'esm2').

Outputs:
- A CSV file ('test_labeled.csv') containing the test set predictions.
"""

import os
import argparse
import _pickle as cPickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def read_seq_pickle(embedding_path: str, seqid: str):
    """
    Loads a single sequence's embedding and label from its pickle file.

    It is assumed that the pickle file was created by writing the embedding array
    first, followed by the label array.

    Args:
        embedding_path (str): The full path to the pickle file (including filename).
        seqid (str): The sequence identifier (used for error messages).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x (np.ndarray): The numerical embedding vector.
            - y (np.ndarray): The corresponding label (e.g., [0.] or [1.]).
    """
    # 'rb' mode is essential for reading binary pickle files.
    with open(embedding_path, "rb") as file:
        try:
            x = cPickle.load(file)
            y = cPickle.load(file)
        except EOFError:
            print(f"Error: Could not read two objects from pickle file for seqid: {seqid}")
            print(f"File path: {embedding_path}")
            # Return None or raise an exception to handle the error upstream
            raise
    return x, y

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

def read_dataset(fasta_filepath, embedding_dir):
    """
    Reads a FASTA file and returns the sequence names, embedded sequences, 
    and labels as 3 separate lists.

    Args:
        fasta_filepath (str): Path to the FASTA file (.fa) for the dataset split
                              (e.g., 'train.fa' or 'test.fa').
        embedding_dir (str): The directory where the embedding .pickle files are stored.

    Returns:
        Tuple[List[str], np.ndarray, np.ndarray]: A tuple containing:
            - seqids (List[str]): A list of sequence identifiers.
            - x (np.ndarray): A 2D NumPy array of the embeddings (N_samples, D_features).
            - y (np.ndarray): A 1D NumPy array of the labels (N_samples,).

    """
    print(f"Loading dataset from: {fasta_filepath}")
    seqs = read_fasta(fasta_filepath)
    seqids = list(seqs.keys())
    x = []
    y = []
    for seqid in tqdm(seqids, desc=f"Loading embeddings from {embedding_dir}"):
        embedding_path = os.path.join(embedding_dir, seqid + ".pickle")
        x_embed, y_label = read_seq_pickle(embedding_path, seqid)
        x.append(x_embed)
        y.append(y_label)

    x = np.array(x).squeeze()
    y = np.array(y).squeeze()
    return seqid, x, y


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script with default values for arguments.")
    # Define arguments and their default values
    parser.add_argument("--training_directory", type=str, default='1_train_splits_fa/ex1', help='e.g. 1_train_splits_fa/ex1')
    parser.add_argument("--embedding_directory", type=str, default='0_embeddings/avgpool_ohe', help="Directory containing the pre-computed .pickle embeddings.")
    parser.add_argument("--embed", type=str, default='ohe', help="One of {'ohe', 'bl62', 'esm2', 'prott5', 'xtrimopglm_10b', 'xtrimopglm_100b'}.")
    parser.add_argument("--output_directory", type=str, default='2_results/ex1', help="e.g. 2_results/ex1")

    args = parser.parse_args()
    training_directory = args.training_directory
    embedding_dir = args.embedding_directory
    embed = args.embed
    output_directory = args.output_directory

    # Load Training and Test Datasets
    seqid_train, x_train, y_train = read_dataset(os.path.join(training_directory, 'train.fa'), embedding_dir)
    seqid_test, x_test, y_test = read_dataset(os.path.join(training_directory, 'test.fa'), embedding_dir)

    # Train the model
    np.random.seed(42)
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=4, random_state = 42)
    clf.fit(x_train, y_train)

    # Get predictions
    y_test_hat = clf.predict_proba(x_test)[:,1]

    # Save predictions
    test_df = pd.DataFrame({"seqid":seqid_test,"y_hat":np.squeeze(y_test_hat), "y":np.squeeze(y_test)})
    os.makedirs(os.path.join(output_directory, embed), exist_ok=True)
    test_df.to_csv(os.path.join(output_directory, embed, "test_labeled.csv"))

    print(f"\nTraining complete. Test results saved to: {os.path.join(output_directory, embed)}")
