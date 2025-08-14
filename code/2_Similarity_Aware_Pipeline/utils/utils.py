import _pickle as cPickle
import numpy as np
import h5py
from operator import itemgetter
import copy
import random
from tqdm import tqdm
from collections import Counter, defaultdict

############################################################
#               Step 1. Get positive splits
############################################################

def get_positive_splits(pos_keys, 
        seqdb, 
        k: int=3):
    """
    Partitions a list of positive keys into k-folds using hierarchical clustering.

    This function groups sequences based on their pairwise identity scores. It starts
    by treating each sequence as its own cluster. It then iteratively merges the
    most similar clusters until the desired number of splits (k) is reached.
    To promote balanced splits, a size restriction is applied to prevent
    clusters from growing too large.

    The algorithm is as follows:
    1. Calculate all pairwise identity scores for the given positive keys.
    2. Sort these pairs in descending order of identity score.
    3. Initialize each key into its own cluster (split).
    4. Calculate an expected maximum size for each split.
    5. Iteratively merge the pair of clusters with the highest identity, provided
       the merged cluster does not exceed the maximum size.
    6. The process stops when the number of clusters is less than or equal to k.

    Args:
        pos_keys: A list of sequence identifiers (strings) to be partitioned.
        seqdb: An instance of the SeqDB class, used to retrieve identity scores
               via its `get_ident` method.
        k: The target number of splits or folds for the partitioning. Defaults to 3.

    Returns:
        A dictionary where each key is a representative sequence ID for a cluster,
        and the value is a set of all sequence IDs belonging to that cluster.
    """
    def join_split(splits, 
            split_memberships, 
            id1, 
            id2, 
            size_restrict
            ):
        """
        Helper function to merge two splits if they are different and within size limits.

        Args:
            splits: The current dictionary of splits {parent_id: {members}}.
            split_memberships: A dict mapping each member to its parent_id.
            id1: The first sequence ID.
            id2: The second sequence ID.
            size_restrict: The maximum allowed size for a merged split.

        Returns:
            A tuple containing the updated splits and split_memberships dictionaries.
        """
        parent_id1, parent_id2 = split_memberships[id1], split_memberships[id2]
        if parent_id1 == parent_id2:
            return splits, split_memberships # Already in the same cluster

        # Ensure consistent ordering for merging
        parent_id1, parent_id2 = sorted((parent_id1, parent_id2))
        new_split = splits[parent_id1].union(splits[parent_id2])
        # Merge only if the new split size is within the restriction
        if len(new_split) <= size_restrict:
            splits[parent_id1] = new_split
            # Update memberships for all elements from the merged split
            for member in splits[parent_id2]:
                split_memberships[member] = parent_id1
            # Remove the old, now empty split
            del splits[parent_id2]
        return splits, split_memberships

    # 1. Get a list of all pairwise identity scores
    all_pairwise = []
    N = len(pos_keys)
    for idx1, k1 in enumerate(pos_keys):
        for idx2, k2 in enumerate(pos_keys[:idx1]):
            identity_score = seqdb.get_ident(k1, k2)
            all_pairwise.append((k1, k2, identity_score))
    
    # 2. Sort pairs by identity score in descending order
    all_pairwise = sorted(all_pairwise, key=itemgetter(-1), reverse=True)

    # 3. Initialize splits, with each key in its own split
    splits: SplitsDict = {k1: {k1} for k1 in pos_keys}
    split_memberships: MembershipDict = {k1: k1 for k1 in pos_keys}
    
    # 4. Calculate the expected (maximum) size of each split to encourage balance
    expected_n = int(np.ceil(N / k))

    # 5. Partitioning: iteratively merge the most similar pairs
    while len(all_pairwise) > 0 and len(splits) > k:
        pair = all_pairwise.pop(0)  # Get the pair with the highest identity
        idx1, idx2 = pair[:2]
        splits, split_memberships = join_split(splits, split_memberships, idx1, idx2, expected_n)
        
    return splits

############################################################
#               Step 2. Remove violations
############################################################

def _get_inter_split_violations(splits, 
        seqdb, 
        threshold: float=0.4
        ):
    """
    Identifies pairs of sequences from different splits that violate an identity threshold.

    This function iterates through all pairs of splits, calculates the pairwise
    identity for all sequences between them, and flags pairs exceeding the
    `threshold`. It also counts how many violations each sequence is involved in.

    Args:
        splits: A dictionary where keys are representative IDs and values are sets of
                sequence IDs belonging to that split.
        seqdb: An instance of the SeqDB class to retrieve identity scores.
        threshold: The sequence identity threshold. Pairs with an identity score
                      above this value are considered violations.

    Returns:
        A list of tuples, where each tuple contains:
        (sequence_1, sequence_2, violations_for_s1, violations_for_s2, total_violations).
        The list is sorted in descending order by the total number of violations.
    """
    splits_keys = sorted(list(splits.keys()))
    violation_pairs = []
    all_violation_counts = []

    for nk1, k1 in enumerate(tqdm(splits_keys, desc="Finding Violations")):
        split1_keys = sorted(list(splits[k1]))
        for k2 in splits_keys[:nk1]:
            split2_keys = sorted(list(splits[k2]))

            # Get the identity matrix for sequences between the two splits
            all_pairwise_ident = seqdb.get_ident(split1_keys, split2_keys)
            
            # Find indices where identity > threshold
            violation_pair_idx = np.array(np.where(all_pairwise_ident > threshold)).T
            
            for vp1, vp2 in violation_pair_idx:
                all_violation_counts.append(split1_keys[vp1])
                all_violation_counts.append(split2_keys[vp2])
                violation_pairs.append((split1_keys[vp1], split2_keys[vp2]))

    violation_counter = Counter(all_violation_counts)
    
    final_pairs = []
    for s1, s2 in tqdm(violation_pairs, desc="Formatting Violations"):
        v_count1 = violation_counter.get(s1, 0)
        v_count2 = violation_counter.get(s2, 0)
        final_pairs.append((s1, s2, v_count1, v_count2, v_count1 + v_count2))
    
    # Sort by the total number of violations for the pair, descending
    final_pairs = sorted(final_pairs, key=itemgetter(-1), reverse=True)
    return final_pairs

def check_violate(splits, 
        seqdb, 
        target_splits, 
        s1, 
        threshold: float = 0.4
        ):
    """
    Wrapper function to check if a sequence `s1` violates the identity threshold
    with any sequence in a target split or collection of splits.

    Args:
        splits: The dictionary of all splits.
        seqdb: The sequence database object.
        target_splits: A single split key (str) or a collection of them (list, set, tuple).
        s1: The sequence ID to check.
        threshold: The identity threshold for a violation.

    Returns:
        True if a violation is found in any of the specified splits, otherwise False.
    """
    if isinstance(target_splits, str):
        # Handle the case of a single target split
        return _check_violate(splits, seqdb, target_splits, s1, threshold)
    elif isinstance(target_splits, (list, tuple, set)):
        # Handle a collection of splits to check against
        for split_key in target_splits:
            if _check_violate(splits, seqdb, split_key, s1, threshold):
                # If a violation is found in any split, we can stop and return True
                return True
        # If the loop completes without finding any violations
        return False
    # Return False for any other unsupported type
    return False

# This helper function is correct and remains unchanged.
def _check_violate(splits, 
        seqdb, 
        target_split, 
        s1, 
        threshold: float=0.4
        ):
    """
    Checks if a sequence `s1` has an identity score above `threshold` with any
    sequence in the specified `target_split`.
    """
    for s2 in splits[target_split]:
        if s1 == s2: continue # A sequence cannot violate with itself
        ident = seqdb.get_ident(s1, s2)
        if ident > threshold:
            return True
    return False

def remove_violations(splits, 
        seqdb, 
        threshold: float=0.4
        ):
    """
    Resolves inter-split violations by removing and then attempting to re-add sequences.

    This function implements a two-phase process:
    1.  **Removal Phase**: It identifies all sequence pairs that violate the identity
        threshold between splits. It then iteratively removes one sequence from each
        violating pair. The sequence from the larger split is chosen for removal to
        help balance split sizes.
    2.  **Add-Back Phase**: It attempts to re-insert the removed sequences. Starting
        with the sequences that had the fewest violations, it checks if adding a
        sequence back to its original split would create a new violation. If not,
        the sequence is re-added.

    Args:
        splits: The initial dictionary of splits with potential violations.
        seqdb: The sequence database object.
        threshold: The identity threshold for defining a violation.

    Returns:
        A new dictionary of splits with violations resolved.
    """
    new_splits = copy.deepcopy(splits)
    split_memberships = {
        seq: k for k, v in new_splits.items() for seq in v
    }

    violation_pairs = _get_inter_split_violations(new_splits, seqdb, threshold)
    
    removed_sequences = set()
    removed_sequences_list = []

    # 1. --- Removal Phase ---
    while len(violation_pairs) > 0:
        pair = violation_pairs.pop(0)
        s1, s2 = pair[0], pair[1]
        # if any one sequence is removed then this pair no longer has to be delt with
        if s1 not in removed_sequences and s2 not in removed_sequences:   
            split_id1, split_id2 = split_memberships[s1], split_memberships[s2]

            # Remove the sequence from the larger split to promote balance
            if len(new_splits[split_id1]) > len(new_splits[split_id2]):
                new_splits[split_id1].remove(s1)
                removed_sequences.add(s1)
                removed_sequences_list.append((s1, split_id1, pair[2]))
            else:
                new_splits[split_id2].remove(s2)
                removed_sequences.add(s2)
                removed_sequences_list.append((s2, split_id2, pair[3]))

    # 2. --- Add-Back Phase ---
    # Adds removed sequences back to the clusters starting from sequences with the least amount of violations
    removed_sequences_list = sorted(removed_sequences_list, key=itemgetter(-1), reverse=False)
    
    for s1, split_id, _ in tqdm(removed_sequences_list, desc="Adding Back Sequences"):
        # Check if adding s1 back to its original cluster creates a new violation
        other_split_keys = list(new_splits.keys())
        if split_id in other_split_keys:
            other_split_keys.remove(split_id)
            
        is_violation = check_violate(new_splits, seqdb, other_split_keys, s1, threshold)
        if not is_violation:
            new_splits[split_id].add(s1)
            
    return new_splits

############################################################
#               Step 3. Get negative splits
############################################################

def get_negative_splits_seeded(pos_splits, 
        neg_splits_base, 
        seqdb, 
        ts, 
        neg_keys, 
        tc: float=0.5, 
        greaterthanorequalto=True, 
        check_violations_at_the_end=True
        ):
    """
    Assigns negative sequences to splits seeded by a positive split structure.

    This function distributes a list of negative sequences into a set of splits that
    mirrors the provided positive splits. This is useful for creating balanced
    k-folds for machine learning, where each fold needs corresponding negative examples.

    The function operates in two main modes based on `tc`:
    1.  **Identity-Based Assignment (`tc > 0`):**
        - It identifies potential pairings between negative and positive sequences
          based on an identity score >= `tc`.
        - It prioritizes assigning negative sequences that have the fewest possible
          positive splits they can be paired with.
        - It assigns negatives to the smallest available valid split to promote balance.
    2.  **Random Assignment (`tc <= 0`):**
        - It randomly shuffles the negative keys and distributes them as evenly
          as possible among the available splits.

    Finally, if `check_violations_at_the_end` is True, it runs a cleanup step to
    remove sequences that cause inter-split identity violations above `ts`.

    Args:
        pos_splits: The pre-defined splits for positive sequences.
        neg_splits_base: A base dictionary for negative splits. Can be empty or
                         partially filled. New negatives will be added to this.
        seqdb: An instance of the SeqDB class for retrieving identity scores.
        ts: The identity threshold to define violations *between*
                           different negative splits.
        neg_keys: A list of all available negative sequence keys to be assigned.
        tc: The identity threshold for pairing a negative sequence with a
                        positive one. If <= 0, random assignment is used.
        greaterthanorequalto: A boolean flag that is currently declared but not used.
        check_violations_at_the_end: If True, runs `remove_violations` after
                                     assignment. If False (and in identity mode),
                                     checks for violations during assignment.

    Returns:
        A dictionary of negative splits corresponding to the positive splits.
    """
    # --- Initialization ---
    # Create a mapping from each positive key to its split ID
    pos_keys = set()
    pos_membership = {}
    for k, v in pos_splits.items():
        for j in v:
            pos_membership[j] = k
            pos_keys.add(j)
    pos_keys = sorted(list(pos_keys))

    # Initialize negative keys
    neg_keys = sorted(list(neg_keys))
    all_neg_keys = neg_keys.copy()
    used_negs = set()
    for k, v in neg_splits_base.items():
        used_negs = used_negs.union(v)
    neg_keys = sorted(list(set(neg_keys).difference(used_negs)))
    neg_splits = copy.deepcopy(neg_splits_base)
    if len(neg_splits) == 0:
        neg_splits = {k: set() for k in pos_splits.keys()}

    # --- Assignment Strategy ---
    if tc > 0:
        ## Get neg clusters ##
        # pairs first by least amout of pairs
        ident_mat = seqdb.get_ident(neg_keys, pos_keys)
        pairs_idx = np.array(np.where(ident_mat >= tc)).T
        all_neg_pairs = {k: [] for k in neg_keys}
        for pidx1, pidx2 in pairs_idx:
            all_neg_pairs[neg_keys[pidx1]].append((neg_keys[pidx1], pos_keys[pidx2], ident_mat[pidx1, pidx2]))
        all_neg_pairs = {k: sorted(v, key=itemgetter(-1), reverse=True) for k, v in all_neg_pairs.items()}
        all_neg_by_pair_count = [(k, len(v)) for k, v in all_neg_pairs.items()]
        all_neg_by_pair_count = sorted(all_neg_by_pair_count, key=itemgetter(-1), reverse=False)

        # Start assigning negatives to positives starting with those that can be paired to the least number of positives
        for (n, c) in all_neg_by_pair_count:
            if c > 0:
                pairs = all_neg_pairs[n]
                pairable_splits = sorted(list(set([pos_membership[p[1]] for p in pairs])))
                pairable_splits = sorted([(p, len(neg_splits[p])) for p in pairable_splits], key=itemgetter(-1), reverse=False)
                if not check_violations_at_the_end:
                    added = False
                    while added == False and len(pairable_splits) > 0:
                        target = pairable_splits.pop(0)
                        target = target[0]
                        violate = False
                        other_splits = list(set(neg_splits.keys()).difference(set([target])))
                        if not check_violate(neg_splits, seqdb, other_splits, n, ts):
                            neg_splits[target].add(n)
                            added = True
                else:
                    neg_splits[pairable_splits[0][0]].add(n)
    else:
        # Random Assignment
        check_violations_at_the_end = True
        random.seed(42)
        random.shuffle(neg_keys)
        k = len(neg_splits.keys())
        split_size = int(np.ceil(len(neg_keys)/k))
        for nk, (k, v) in enumerate(neg_splits.items()):
            neg_splits[k] = neg_splits[k].union(set(neg_keys[nk*split_size:(nk+1)*split_size]))

    if check_violations_at_the_end:
        neg_splits = remove_violations(neg_splits, seqdb, ts)

    return neg_splits

############################################################
#           Step 4. Dataset construction strategy
############################################################

def get_length_distribution(list_of_seq, binsize: int = 50):
    """
    Calculates a binned probability distribution of sequence lengths.

    This function takes a list of sequences, calculates their lengths, and groups
    them into bins of a specified size. It then computes the probability of a
    sequence length falling into each bin. A small probability is added to all
    bins to prevent any bin from having zero probability (smoothing), which can be
    useful for sampling.

    Args:
        list_of_seq: A list of sequences (strings).
        binsize: The size of each length bin for the histogram.

    Returns:
        A defaultdict where keys are the starting integer of each length bin and
        values are the corresponding probabilities (float).
    """
    if not list_of_seq:
        return defaultdict(float)

    list_of_seq_lengths: List[int] = [len(i) for i in list_of_seq]
    max_bin = ((np.max(list_of_seq_lengths) // binsize) + 1) * binsize
    
    counts, bins = np.histogram(list_of_seq_lengths, bins=np.arange(0, max_bin + binsize, binsize))
    
    # Convert counts to a probability distribution
    prob: np.ndarray = counts / np.sum(counts)
    # Add a small probability to each bin to avoid zeros (smoothing)
    prob += 0.01
    # Re-normalize so the probabilities sum to 1
    prob /= np.sum(prob)
    
    prob_dict: ProbDict = defaultdict(float)
    for n, b in enumerate(bins[:-1]):
        prob_dict[b] = prob[n]
        
    return prob_dict

def get_length_control(
        pos_splits,
        neg_splits,
        seqdb,
        binsize: int = 50,
        randomseed: int = 42
        ):
    """
    Subsamples negative splits to match the length distribution of positive splits.

    For each split, this function compares the negative sequences to the positive ones.
    If there are more negatives than positives, it subsamples the negatives so that
    their length distribution matches that of the positives. If, after this process,
    the number of negatives is still less than the positives, it "tops up" the set
    by randomly sampling from the remaining negatives.

    Args:
        pos_splits: A dictionary mapping split IDs to sets of positive sequence keys.
        neg_splits: A dictionary mapping split IDs to sets of negative sequence keys.
        seqdb: An instance of the SeqDB class to retrieve sequence data.
        binsize: The bin size to use for calculating length distributions.
        randomseed: A seed for the random number generator for reproducibility.

    Returns:
        A defaultdict containing the new, length-controlled negative splits.
    """
    max_len = 1000  # Sequences longer than this are disqualified
    balanced_neg_splits = defaultdict(set)
    np.random.seed(randomseed)

    for split in sorted(list(pos_splits.keys())):
        pos_keys = pos_splits[split]
        neg_keys = neg_splits[split]
        N_pos = len(pos_keys)

        if len(neg_keys) > N_pos:
            # --- Perform length control only if there are excess negatives ---
            pos_seqs = [seqdb.get_seq(key) for key in pos_keys]
            
            # Get the target length distribution from the positive sequences
            probs: ProbDict = get_length_distribution(pos_seqs, binsize=binsize)
            
            # Bin the negative sequences by length
            neg_seqs_in_bins = defaultdict(list)
            for neg_key in neg_keys:
                neg_seq = seqdb.get_seq(neg_key)
                if len(neg_seq) < max_len:
                    bin_start = (len(neg_seq) // binsize) * binsize
                    neg_seqs_in_bins[bin_start].append(neg_key)

            # Sample from each bin according to the positive distribution
            for length_bin, neg_seqs_in_bin in neg_seqs_in_bins.items():
                if len(neg_seqs_in_bin) > 0:
                    p = probs[length_bin]
                    n_to_sample = int(N_pos * p)
                    
                    shuffled_negs = np.random.permutation(neg_seqs_in_bin)
                    # Take at most the number of available negatives in the bin
                    n_final = min(len(shuffled_negs), n_to_sample)
                    balanced_neg_splits[split].update(shuffled_negs[:n_final])
            
            # --- Top-up if necessary ---
            # If length-based sampling resulted in fewer negatives than positives, add more randomly.
            N_neg = len(balanced_neg_splits[split])
            if N_neg < N_pos:
                topup_count = N_pos - N_neg
                remaining_neg_keys = neg_keys.difference(balanced_neg_splits[split])
                shuffled_remaining = np.random.permutation(list(remaining_neg_keys))
                balanced_neg_splits[split].update(shuffled_remaining[:topup_count])
        else:
            # If there are not enough negatives to subsample, keep them all
            balanced_neg_splits[split] = neg_keys
            
    return balanced_neg_splits

def get_balance(
        pos_splits,
        neg_splits,
        randomseed: int = 42
        ):
    """
    Balances splits by randomly downsampling negatives to match the number of positives.

    This function provides a simple way to balance splits. For each split, if the
    number of negative sequences is greater than the number of positive sequences,
    it will randomly select a subset of the negatives to equal the number of
    positives. Unlike `control_for_length`, this function does not consider
    sequence length.

    Note:
        - The `seqdb` argument is included for API consistency but is not used.
        - The implementation has been corrected to ensure all splits are returned,
          not just those that were downsampled.

    Args:
        pos_splits: A dictionary mapping split IDs to sets of positive sequence keys.
        neg_splits: A dictionary mapping split IDs to sets of negative sequence keys.
        seqdb: An instance of the SeqDB class (currently unused).
        randomseed: A seed for the random number generator for reproducibility.

    Returns:
        A dictionary containing the balanced negative splits.
    """
    balanced_neg_splits = {}
    np.random.seed(randomseed)
    
    for split, pos_keys in pos_splits.items():
        N_pos = len(pos_keys)
        neg_keys_this_split = list(neg_splits.get(split, set()))
        
        if len(neg_keys_this_split) > N_pos:
            # If there are more negatives, randomly downsample them
            shuffled_neg_keys = np.random.permutation(neg_keys_this_split)
            balanced_neg_splits[split] = set(shuffled_neg_keys[:N_pos])
        else:
            # Otherwise, keep all the existing negatives for that split
            balanced_neg_splits[split] = set(neg_keys_this_split)
            
    return balanced_neg_splits

def get_minimal(
        pos_splits,
        neg_splits,
        seqdb, 
        min_pos, 
        min_neg, 
        tc, 
        randomseed=42):
    """
    Subsamples positive and negative splits to create minimal, representative sets.

    This function iterates through each data split and reduces the number of positive
    and negative sequences to `min_pos` and `min_neg`, respectively. It operates
    in two modes based on the identity threshold `tc`:

    1.  **Random Sampling (`tc <= 0`):**
        Performs a simple random selection of sequences for each split.

    2.  **Identity-Based Sampling (`tc > 0`):**
        Ensures that the selected positive and negative sequences have some
        similarity. It first selects `min_pos` positives, prioritizing those that
        have at least one negative partner above the `tc` identity threshold. It then
        selects `min_neg` negatives that have at least one partner among the
        newly selected positives.

    Args:
        pos_splits: A dictionary mapping split IDs to collections of positive sequence keys.
        neg_splits: A dictionary mapping split IDs to collections of negative sequence keys.
        seqdb: An instance of the SeqDB class for retrieving identity scores.
        min_pos: The target number of positive sequences for each new split.
        min_neg: The target number of negative sequences for each new split.
        tc: The sequence identity threshold. If > 0, identity-based sampling is used.
        randomseed: A seed for the random number generator for reproducibility.

    Returns:
        Two dictionary containing the new, minimal positive and negative splits.
    """
    min_pos_splits = {}
    min_neg_splits = {}

    for split in pos_splits.keys():
        all_seqs = {'pos': list(pos_splits[split]), 'neg': list(neg_splits[split])}

        if tc <= 0:
            subset_pos = list(np.random.choice(all_seqs['pos'], min_pos, replace=False))
            subset_neg = list(np.random.choice(all_seqs['neg'], min_neg, replace=False))
            min_pos_splits[split] = subset_pos
            min_neg_splits[split] = subset_neg
        else: 
            # Find all positives that share > Tc with negatives and subset
            all_neg_h5 = seqdb.get_hdf5_mapped_idx(all_seqs['neg'])
            possible_pos = []
            for i in all_seqs['pos']:
                ident_list = seqdb.get_ident_list(i)[all_neg_h5]
                neg_idx = np.where(ident_list >= tc)[0]
                if len(neg_idx) > 0:
                    possible_pos.append(i)
            other_pos = list(set(all_seqs['pos']).difference(set(possible_pos)))
            n = min_pos-len(possible_pos)
            if n > 0:
                np.random.seed(randomseed)
                subset_pos = possible_pos + list(np.random.choice(other_pos, n, replace=False))
            elif n < 0:
                np.random.seed(randomseed)
                subset_pos = list(np.random.choice(possible_pos, min_pos, replace=False))
            else:
                subset_pos = possible_pos

            subset_pos_h5 = seqdb.get_hdf5_mapped_idx(subset_pos)

            # randomly subset out negatives
            subset_neg = []
            all_negs = all_seqs['neg'].copy()
            np.random.seed(randomseed)
            np.random.shuffle(all_negs)

            while len(subset_neg) < min_neg and len(all_negs) > 0:
                i = all_negs.pop()
                ident_list = seqdb.get_ident_list(i)[subset_pos_h5]
                pos_idx = np.where(ident_list >= tc)[0]
                if len(pos_idx) > 0:    # only retain negatives that share > Tc with at least one positive
                    subset_neg.append(i)
            if len(subset_neg) < min_neg:
                print("Negatives are less than the specified minimum: {:} < {:}".format(len(subset_neg), min_neg))
            if len(subset_pos) < min_pos:
                print("Positives are less than the specified minimum: {:} < {:}".format(len(subset_pos), min_pos))
            min_pos_splits[split] = subset_pos
            min_neg_splits[split] = subset_neg
    return min_pos_splits, min_neg_splits