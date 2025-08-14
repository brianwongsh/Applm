import _pickle as cPickle
import numpy as np
import h5py
from operator import itemgetter
import copy
import random
from tqdm import tqdm
from collections import Counter, defaultdict, OrderedDict
import torch
import esm
from transformers import T5Tokenizer, T5EncoderModel
import re
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig

class OHE_Encoder():
    """
    Encodes amino acid sequences using one-hot encoding followed by average pooling.

    Attributes:
        AA (List[str]): A list of the 20 standard amino acids plus 'X' for unknown/other.
        AA_DICT (defaultdict[str, int]): A mapping from each amino acid character to an
                                         integer index.
    """
    def __init__(self):
        """
        Initializes the encoder with a standard amino acid vocabulary and mapping.

        The vocabulary includes 20 standard amino acids and an 'X' character, which
        serves as the default for any non-standard characters encountered in a
        sequence, thanks to the use of `defaultdict`.
        """
        self.AA = ['X','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        self.AA_DICT = defaultdict(int) # defaultdict because any non-existing keys will be mapped to index 0 i.e. X
        for i, aa in enumerate(self.AA):
            self.AA_DICT[aa] = i

    def encode_sequence(self, single_sequence: str):
        """
        Converts a single amino acid sequence into a fixed-size vector.

        Args:
            single_sequence: A string representing the amino acid sequence.

        Returns:
            A 2D numpy array of shape (21, 1) representing the average pooled
            one-hot encoding of the sequence.
        """
        single_sequence = single_sequence.upper()
        vocab_size = len(self.AA)
        seq_length = len(single_sequence)

        # Create a base matrix for the one-hot encoding
        base = np.zeros((vocab_size, seq_length))
        
        # Populate the matrix: for each position in the sequence, set the
        # corresponding amino acid's index to 1.
        for n_aa, aa in enumerate(single_sequence):
            aa_index = self.AA_DICT[aa]
            base[aa_index, n_aa] = 1
            
        # Apply average pooling across the sequence length axis (axis=1).
        # This calculates the frequency of each amino acid.
        # `keepdims=True` ensures the output shape is (21, 1), not (21,).
        pooled_encoding = np.mean(base, axis=1, keepdims=True)
        
        return pooled_encoding

class BL62_Encoder():
    """
    Encodes amino acid sequences using BLOSUM62 substitution scores.

    Attributes:
        blosum_dict (Dict[str, List[float]]): A dictionary mapping each amino acid
            (and ambiguous characters B, Z, X) to its BLOSUM62 substitution scores.
    """
    def __init__(self):
        """
        Initializes the encoder with a hardcoded BLOSUM62 substitution matrix.
        """
        self.blosum_dict = {'A': [4.0, -1.0, -2.0, -2.0, 0.0, -1.0, -1.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 1.0, 0.0, -3.0, -2.0, 0.0, -2.0, -1.0, 0.0],
                            'R': [-1.0, 5.0, 0.0, -2.0, -3.0, 1.0, 0.0, -2.0, 0.0, -3.0, -2.0, 2.0, -1.0, -3.0, -2.0, -1.0, -1.0, -3.0, -2.0, -3.0, -1.0, 0.0, -1.0],
                            'N': [-2.0, 0.0, 6.0, 1.0, -3.0, 0.0, 0.0, 0.0, 1.0, -3.0, -3.0, 0.0, -2.0, -3.0, -2.0, 1.0, 0.0, -4.0, -2.0, -3.0, 3.0, 0.0, -1.0],
                            'D': [-2.0, -2.0, 1.0, 6.0, -3.0, 0.0, 2.0, -1.0, -1.0, -3.0, -4.0, -1.0, -3.0, -3.0, -1.0, 0.0, -1.0, -4.0, -3.0, -3.0, 4.0, 1.0, -1.0],
                            'C': [0.0, -3.0, -3.0, -3.0, 9.0, -3.0, -4.0, -3.0, -3.0, -1.0, -1.0, -3.0, -1.0, -2.0, -3.0, -1.0, -1.0, -2.0, -2.0, -1.0, -3.0, -3.0, -2.0],
                            'Q': [-1.0, 1.0, 0.0, 0.0, -3.0, 5.0, 2.0, -2.0, 0.0, -3.0, -2.0, 1.0, 0.0, -3.0, -1.0, 0.0, -1.0, -2.0, -1.0, -2.0, 0.0, 3.0, -1.0],
                            'E': [-1.0, 0.0, 0.0, 2.0, -4.0, 2.0, 5.0, -2.0, 0.0, -3.0, -3.0, 1.0, -2.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0, 1.0, 4.0, -1.0],
                            'G': [0.0, -2.0, 0.0, -1.0, -3.0, -2.0, -2.0, 6.0, -2.0, -4.0, -4.0, -2.0, -3.0, -3.0, -2.0, 0.0, -2.0, -2.0, -3.0, -3.0, -1.0, -2.0, -1.0],
                            'H': [-2.0, 0.0, 1.0, -1.0, -3.0, 0.0, 0.0, -2.0, 8.0, -3.0, -3.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -2.0, 2.0, -3.0, 0.0, 0.0, -1.0],
                            'I': [-1.0, -3.0, -3.0, -3.0, -1.0, -3.0, -3.0, -4.0, -3.0, 4.0, 2.0, -3.0, 1.0, 0.0, -3.0, -2.0, -1.0, -3.0, -1.0, 3.0, -3.0, -3.0, -1.0],
                            'L': [-1.0, -2.0, -3.0, -4.0, -1.0, -2.0, -3.0, -4.0, -3.0, 2.0, 4.0, -2.0, 2.0, 0.0, -3.0, -2.0, -1.0, -2.0, -1.0, 1.0, -4.0, -3.0, -1.0],
                            'K': [-1.0, 2.0, 0.0, -1.0, -3.0, 1.0, 1.0, -2.0, -1.0, -3.0, -2.0, 5.0, -1.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0, 0.0, 1.0, -1.0],
                            'M': [-1.0, -1.0, -2.0, -3.0, -1.0, 0.0, -2.0, -3.0, -2.0, 1.0, 2.0, -1.0, 5.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, 1.0, -3.0, -1.0, -1.0],
                            'F': [-2.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -1.0, 0.0, 0.0, -3.0, 0.0, 6.0, -4.0, -2.0, -2.0, 1.0, 3.0, -1.0, -3.0, -3.0, -1.0],
                            'P': [-1.0, -2.0, -2.0, -1.0, -3.0, -1.0, -1.0, -2.0, -2.0, -3.0, -3.0, -1.0, -2.0, -4.0, 7.0, -1.0, -1.0, -4.0, -3.0, -2.0, -2.0, -1.0, -2.0],
                            'S': [1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -2.0, 0.0, -1.0, -2.0, -1.0, 4.0, 1.0, -3.0, -2.0, -2.0, 0.0, 0.0, 0.0],
                            'T': [0.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 1.0, 5.0, -2.0, -2.0, 0.0, -1.0, -1.0, 0.0],
                            'W': [-3.0, -3.0, -4.0, -4.0, -2.0, -2.0, -3.0, -2.0, -2.0, -3.0, -2.0, -3.0, -1.0, 1.0, -4.0, -3.0, -2.0, 11.0, 2.0, -3.0, -4.0, -3.0, -2.0],
                            'Y': [-2.0, -2.0, -2.0, -3.0, -2.0, -1.0, -2.0, -3.0, 2.0, -1.0, -1.0, -2.0, -1.0, 3.0, -3.0, -2.0, -2.0, 2.0, 7.0, -1.0, -3.0, -2.0, -1.0],
                            'V': [0.0, -3.0, -3.0, -3.0, -1.0, -2.0, -2.0, -3.0, -3.0, 3.0, 1.0, -2.0, 1.0, -1.0, -2.0, -2.0, 0.0, -3.0, -1.0, 4.0, -3.0, -2.0, -1.0],
                            'B': [-2.0, -1.0, 3.0, 4.0, -3.0, 0.0, 1.0, -1.0, 0.0, -3.0, -4.0, 0.0, -3.0, -3.0, -2.0, 0.0, -1.0, -4.0, -3.0, -3.0, 4.0, 1.0, -1.0],
                            'Z': [-1.0, 0.0, 0.0, 1.0, -3.0, 3.0, 4.0, -2.0, 0.0, -3.0, -3.0, 1.0, -1.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0, 1.0, 4.0, -1.0],
                            'X': [0.0, -1.0, -1.0, -1.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0, 0.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0]}

    def encode_sequence(self, single_sequence: str):
        """
        Converts a single amino acid sequence into a fixed-size BLOSUM62 vector.

        Args:
            single_sequence: A string representing the amino acid sequence.

        Returns:
            A 2D numpy array of shape (23, 1) representing the average pooled
            BLOSUM62 encoding of the sequence.
        """
        b_embed = []
        for aa in single_sequence.upper():
            # Retrieve the BLOSUM62 vector for the amino acid.
            b_vector = self.blosum_dict[aa]
            # Convert to a column vector and append to the list.
            b_embed.append(np.array(b_vector)[:, np.newaxis])
        
        # Concatenate the list of column vectors into a single matrix of shape (23, seq_length).
        b_embed_matrix = np.concatenate(b_embed, axis=1)
        
        # Apply average pooling across the sequence length axis (axis=1).
        # `keepdims=True` ensures the output shape is (23, 1), not (23,).
        pooled_embedding = np.mean(b_embed_matrix, axis=1, keepdims=True)
        
        return pooled_embedding

class ESM2_Encoder():
    """
    Encodes amino acid sequences using the pre-trained ESM-2 protein language model.

    Attributes:
        device (torch.device): The computing device (CUDA or CPU) used for the model.
        repr_layer (int): The layer from which representations are extracted (33 for this model).
        batch_converter (callable): A function to convert sequence data into model-readable tokens.
        model (esm.ESM2): The loaded and prepared ESM-2 model, set to evaluation mode.
    """
    def __init__(self):
        """
        Initializes the encoder by loading the ESM-2 model and preparing it for inference.
        """
        self.device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load the pre-trained ESM-2 model and its corresponding alphabet
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.repr_layer = 33  # The final layer of this model
        
        # Freeze model parameters as we are only doing inference
        for param in model.parameters():
            param.requires_grad = False
            
        model = model.to(self.device)
        self.batch_converter = alphabet.get_batch_converter()
        # Set the model to evaluation mode for deterministic results
        self.model: esm.ESM2 = model.eval()
    
    def encode_sequence(self, single_sequence: str):
        """
        Converts a single amino acid sequence into a fixed-size ESM-2 embedding.

        Args:
            single_sequence: A string representing the amino acid sequence.

        Returns:
            A 2D numpy array of shape (1280, 1) representing the average-pooled
            ESM-2 embedding of the sequence.
        """
        single_sequence = single_sequence.upper()
        # Clear GPU cache to free up memory before processing
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Format the data for the batch converter
        data = [("placeholder_id", single_sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Run the model in inference mode
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer])
            
        # Extract the representations from the specified layer
        token_representations = results["representations"][self.repr_layer]
        
        # Slice the tensor to remove the special start (<cls>) and end (<eos>) tokens.
        # Shape: (batch_size, seq_length, embedding_dim) -> (1, seq_length-2, 1280)
        sequence_representations = token_representations[:, 1:-1, :]
        
        # Apply average pooling across the sequence length dimension (dim=1).
        # Shape: (1, seq_length-2, 1280) -> (1, 1, 1280)
        pooled_embed = torch.mean(sequence_representations, dim=1, keepdim=True)
        
        # Detach from the graph, move to CPU, and convert to NumPy array.
        # Shape: (1, 1, 1280) -> (1, 1, 1280)
        numpy_embed = pooled_embed.detach().cpu().numpy()
        
        # Reshape to (1, 1280, 1) and then swap axes to get the final (1280, 1) shape.
        final_embed = np.swapaxes(numpy_embed[0], 1, 0)
        
        return final_embed
        
class T5_Encoder():
    """
    Encodes amino acid sequences using the pre-trained ProtT5-XL-U50 model.

    Attributes:
        device (torch.device): The computing device (CUDA or CPU) used for the model.
        model (T5EncoderModel): The loaded and prepared ProtT5 model, set to evaluation mode.
        tokenizer (T5Tokenizer): The tokenizer corresponding to the ProtT5 model.
    """
    def __init__(self):
        """
        Initializes the encoder by loading the ProtT5 model and preparing it for inference.
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
        
        print(f"Loading ProtT5 model: {transformer_link}")
        model = T5EncoderModel.from_pretrained(transformer_link)
        
        # Use half-precision on GPU for faster inference and lower memory usage
        if self.device.type == 'cuda':
            print("Using half precision on GPU.")
            model.half()
        else:
            print("Using full precision on CPU.")
            
        self.model = model.to(self.device).eval()
        self.tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)

    def encode_sequence(self, single_sequence: str) :
        """
        Converts a single amino acid sequence into a fixed-size ProtT5 embedding.

        Args:
            single_sequence: A string representing the amino acid sequence.

        Returns:
            A 2D numpy array of shape (1024, 1) representing the average-pooled
            ProtT5 embedding of the sequence.
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Preprocess the sequence: replace rare AAs with 'X' and add spaces.
        # The model expects space-separated amino acids.
        sequence_preprocessed = " ".join(list(re.sub(r"[UZOB]", "X", single_sequence.upper())))

        # Tokenize the sequence and move tensors to the correct device
        ids: Dict[str, List[int]] = self.tokenizer.batch_encode_plus(
            [sequence_preprocessed], add_special_tokens=True, padding="longest"
        )
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        # Run the model in inference mode
        with torch.no_grad():
            # The model output is a tuple; the first element is the last hidden state.
            embedding_result = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding_result.last_hidden_state

        # The last token is a special <eos> token, which we remove before pooling.
        # Shape: (batch_size, seq_length, embedding_dim) -> (1, seq_length-1, 1024)
        embedding = embedding[0, :-1, :]
        
        # Apply average pooling across the sequence length dimension (dim=0).
        # Shape: (seq_length-1, 1024) -> (1024,)
        embedding = torch.mean(embedding, dim=0)

        # Reshape and convert to NumPy array.
        # Shape: (1024,) -> (1, 1024) -> (1024, 1)
        embedding = embedding.detach().cpu().numpy()
        embedding = np.expand_dims(embedding, axis=0) # Add batch dimension back for swap
        embedding = np.swapaxes(embedding, 1, 0)
            
        return embedding

xtrimopglm100b_device_map = OrderedDict({
    'transformer.embedding': 0, 
    'transformer.encoder.layers.0': 0, 'transformer.encoder.layers.1': 0, 
    'transformer.encoder.layers.2': 0, 'transformer.encoder.layers.3': 0, 
    'transformer.encoder.layers.4': 0, 'transformer.encoder.layers.5': 0, 
    'transformer.encoder.layers.6': 0, 'transformer.encoder.layers.7': 0, 
    'transformer.encoder.layers.8': 0, 'transformer.encoder.layers.9': 0, 
    'transformer.encoder.layers.10': 0, 'transformer.encoder.layers.11': 0, 
    'transformer.encoder.layers.12': 0, 'transformer.encoder.layers.13': 0, 
    'transformer.encoder.layers.14': 0, 'transformer.encoder.layers.15': 0, 
    'transformer.encoder.layers.16': 0, 'transformer.encoder.layers.17': 0, 
    'transformer.encoder.layers.18': 0, 'transformer.encoder.layers.19': 0, 
    'transformer.encoder.layers.20': 0, 'transformer.encoder.layers.21': 0, 
    'transformer.encoder.layers.22': 0, 'transformer.encoder.layers.23': 0, 
    'transformer.encoder.layers.24': 0, 'transformer.encoder.layers.25': 0, 
    'transformer.encoder.layers.26': 0, 'transformer.encoder.layers.27': 0, 
    'transformer.encoder.layers.28': 0, 'transformer.encoder.layers.29': 0, 
    'transformer.encoder.layers.30': 0, 'transformer.encoder.layers.31': 0, 
    'transformer.encoder.layers.32': 0, 'transformer.encoder.layers.33': 0, 
    'transformer.encoder.layers.34': 0, 'transformer.encoder.layers.35': 0, 
    'transformer.encoder.layers.36': 0, 'transformer.encoder.layers.37': 0, 
    'transformer.encoder.layers.38': 0, 'transformer.encoder.layers.39': 0, 
    'transformer.encoder.layers.40': 0, 'transformer.encoder.layers.41': 0, 
    'transformer.encoder.layers.42': 0, 'transformer.encoder.layers.43': 0, 
    'transformer.encoder.layers.44': 0, 'transformer.encoder.layers.45': 0, 
    'transformer.encoder.layers.46': 0, 'transformer.encoder.layers.47': 0, 
    'transformer.encoder.layers.48': 0, 'transformer.encoder.layers.49': 0, 
    'transformer.encoder.layers.50': 0, 'transformer.encoder.layers.51': 1, 
    'transformer.encoder.layers.52': 1, 'transformer.encoder.layers.53': 1, 
    'transformer.encoder.layers.54': 1, 'transformer.encoder.layers.55': 1, 
    'transformer.encoder.layers.56.input_layernorm': 1, 
    'transformer.encoder.layers.56.self_attention': 1, 
    'transformer.encoder.layers.56.post_attention_layernorm': 1, 
    'transformer.encoder.layers.57': 1, 'transformer.encoder.layers.58': 1, 
    'transformer.encoder.layers.59': 1, 'transformer.encoder.layers.60': 1, 
    'transformer.encoder.layers.61': 1, 'transformer.encoder.layers.62': 1, 
    'transformer.encoder.layers.63': 1, 'transformer.encoder.layers.64': 1, 
    'transformer.encoder.layers.65': 1, 'transformer.encoder.layers.66': 1, 
    'transformer.encoder.layers.67': 1, 'transformer.encoder.layers.68': 1, 
    'transformer.encoder.layers.69': 1, 'transformer.encoder.layers.70': 1, 
    'transformer.encoder.layers.71': 1, 'transformer.encoder.final_layernorm': 1, 
    'transformer.output_layer': 1, 'transformer.encoder.layers.56.mlp': 1
})
"""
A device map to distribute the ProteinGLM 100B model across two GPUs.
Layers 0-50 are placed on GPU 0, and layers 51-71 and the final layers
are placed on GPU 1. This is necessary due to model size.
"""

class xTrimoPGLM_Encoder():
    """
    Encodes sequences using xTrimoPGLM models (e.g., 10B or 100B versions).

    This class handles loading different sizes of the xTrimoPGLM models and
    generating embeddings. It has special logic for the 100B model, which
    requires multi-GPU support via a device map.

    Attributes:
        model_name (str): The name of the Hugging Face model being used.
        tokenizer: The tokenizer for the loaded model.
        model: The loaded xTrimoPGLM model, set to evaluation mode.
        device (torch.device): The primary device for the model (for smaller models).
    """
    def __init__(self, model_name: str = "biomap-research/proteinglm-10b-mlm", cuda_device: int = 0):
        """
        Initializes the tokenizer and model.

        If '100b' is in the model name, it loads the quantized 100B model and
        distributes it across multiple GPUs using the predefined `device_map`.
        Otherwise, it loads the specified model onto a single target device.

        Args:
            model_name: The Hugging Face identifier for the model.
            cuda_device: The index of the CUDA device to use for smaller models.
        """
        self.model_name = model_name
        
        if '100b' in model_name:
            # --- Logic for the 100B model (multi-GPU) ---
            self.tokenizer = AutoTokenizer.from_pretrained("biomap-research/proteinglm-100b-int4", trust_remote_code = True, use_fast = True)
            config = AutoConfig.from_pretrained("biomap-research/proteinglm-100b-int4",  trust_remote_code = True, torch_dtype = torch.half)
            config.is_causal = False
            config.post_layer_norm=True # use the final layernorm or not, some tasks set to false would be better.
            self.model = AutoModelForMaskedLM.from_pretrained("biomap-research/proteinglm-100b-int4", 
                config = config, 
                torch_dtype=torch.half, 
                trust_remote_code=True, 
                device_map=xtrimopglm100b_device_map
                )
        else:
            # --- Logic for smaller models (e.g., 10B, single-GPU) ---
            self.device = torch.device('cuda:{:}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')
            self.tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name,  trust_remote_code=True, torch_dtype=torch.bfloat16)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model = self.model.to(self.device)
            self.model = self.model.eval()

        # Freeze parameters and set to evaluation mode for inference
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.eval()
    
    def encode_sequence(self, single_sequence: str):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Tokenize the input sequence
        output = self.tokenizer(single_sequence, add_special_tokens=True, return_tensors='pt')
        with torch.inference_mode():
            inputs = {"input_ids": output["input_ids"].to(self.device), 
                "attention_mask": output["attention_mask"].to(self.device)}
            output_embeddings = self.model(**inputs, output_hidden_states=True, return_last_hidden_state=True).hidden_states[:-1, 0].to(dtype=torch.float32) # get rid of the <eos> token
        embed = output_embeddings.detach().cpu().numpy()
        embed = np.swapaxes(embed, 1, 0)
        embed = np.mean(embed, axis=1, keepdims=True)
        return embed
