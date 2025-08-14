# Driving Accurate Allergen Prediction with Protein Language Models and Generalization-Focused Evaluation

<!-- Optional but recommended: Badges for DOI, license, etc. You can get these from sites like shields.io -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Associated Publication

This repository contains code to train and reproduce the models and the similarity-aware pipeline used in the following publication.

> Wong, et al. (2025). "Driving Accurate Allergen Prediction with Protein Language Models and Generalization-Focused Evaluation". *TBD*, Volume (Issue), pages. [https://doi.org/your/paper/doi](https://doi.org/your/paper/doi)

## Table of Contents

- [Overview](#overview)
- [How to Run the Code](#how-to-run-the-code)
- [System Requirements](#system-requirements)
- [Installation and Dependencies](#installation-and-dependencies)
- [Example Data](#example-data)
- [License](#license)
- [How to Cite](#how-to-cite)
- [Authors and Acknowledgements](#authors-and-acknowledgements)
- [Contact Information](#contact-information)

## Overview
This repository provides the official implementation for Applm, a novel framework for predicting protein allergenicity using embeddings from Protein Language Models. We also introduce a robust similarity-aware evaluation pipeline designed to prevent data leakage and provide more realistic estimates of model generalization. The repository contains the code to reproduce all data splits, model training, and results presented in our paper.

The repository is organized as follows:

- The `code` directory contains code to extract embeddings used for Applm, code to perform similarity-aware splitting as described in the paper, and code to traing and test Applm.

- The `dataset` directory contains the training and test settings used in the publication.

- The `results` directory contains the external and internal benchmarking results shown in the paper.

The basic workflow from the data preprocessing, to dataset splitting, to training and testing the model can be run with the provided example data as follows.

## How to Run the Code

### (Optional) Step 1: Computing the pairwise sequence similarity
Here is an example workflow of using `ssearch36` from the package [`fasta36`](https://github.com/wrpearson/fasta36) to compute all pairwise sequence identities of a given set of sequences as performed in the publication. The example can be followed in `code/1_Computing_Similarity`. This step requires the installation of `fasta36` package. `fasta v36.3.8e` is used for this publication

The `run_ssearch36.sh` file takes as input a fasta file, and computes the pairwise sequence identity of sequences in the fasta file using `ssearch36` from the `fasta36` package. The processed output is saved at the provided output directory.

```
# Executing in the `code/1_Computing_Similarity` directory

./run_ssearch36.sh \
    example_input/ex3.fa \   # Fasta with sequences of interest
    example_output/ex3      # Output directory
```
This step is optional because others may prefer other ways of computing similarity or distance between sequences, as long as you have your computed set of pairwise similarities.

Note that a fasta file of 5000 sequences (`ex3.fa`) using 16 threads took around 127 minutes to finish computing sequence identities. 

### Step 2: Similarity-aware pipeline
The following example demonstrates running the similarity-aware pipeline from the folder `code/2_Similarity_Aware_Pipeline`. First, preprocess the data following `run_preprocess.py`. This format helps us manipulate sequence and identity data during splitting. 

`run_preprocess.py` takes as input the fasta file with sequences and the ssearch36 output produced above, and produces three outputs saved in `--output_directory`, including `<fasta_filename>_pairwise_ident.npy`, `<fasta_filename>_key_to_idx.pickle` and `<fasta_filename>.h5`, which are precomputed data structures to help us quickly access sequence identities during splitting.

If you already have pre-computed pairwise identity data in the format of a 2D numpy array, you can simply run  `run_preprocess_from_matrix.py` with that matrix. This file also produces the same outputs.
```
# Executing in the `code/2_Similarity_Aware_Pipeline` directory

python run_preprocess.py \
    --fasta_path ../1_Computing_Similarity/example_input/ex3.fa \
    --ssearch_output_path ../1_Computing_Similarity/example_output/ex3_formatted \
    --output_directory example_intermediate \
    --coverage_control 0.25

# If you have already precomputed pairwise identites for fastas in the 
# format of an numpy array.
#
# python run_preprocess_from_matrix.py \
#     --fasta_path ../1_Computing_Similarity/example_input/ex3.fa \
#     --matrix_path example_intermediate/ex3_pairwise_ident.npy \
#     --output_directory example_intermediate \
#     --coverage_control 0.25
```
Afterwards, run `run_similarity_aware_pipeline.py` with the outputs from `run_preprocess.py`.
```
# Executing in the `code/2_Similarity_Aware_Pipeline` directory

python run_similarity_aware_pipeline.py \
  --fasta_path ../1_Computing_Similarity/example_input/ex3.fa \
  --keys_to_idx_path example_intermediate/ex3_key_to_idx.pickle \
  --hdf5_path example_intermediate/ex3.h5 \
  --output_directory example_output/partitioned_fasta \
  --ts 0.4 \                        # Ts, Similarity threshold between splits
  --tc 0.5                          # Tc, Similarity threshold between classes
```
The specific splitting function can be found in `code/2_Similarity_Aware_Pipeline/utils/utils.py`. A step-by-step implementation of the pipeline and checking of violations can also be found in `code/2_Similarity_Aware_Pipeline/notebook/1_main_workflow.ipynb`. The implementation of the data structure used can be found in `code/2_Similarity_Aware_Pipeline/utils/sequenceDatabaseObject.py`

### Step 3: Encode sequence
After partitioning, sequences are encoded into embeddings for training and testing by running `run_embed.py` in `code/3_Extracting_Embeddings`.
```
# Executing in the `code/3_Extracting_Embeddings` directory

python run_embed.py \
    --embed ohe \
    --fasta_path example_input/example.fa \
    --output_directory example_output/
```
 The detailed implementation of each embedding method can be found in `code/3_Extracting_Embeddings/utils/encoders.py`. A step-by-step implementation can also be found in `code/3_Extracting_Embeddings/notebook/1_extracting_embeddings.ipynb`.

### Step 4: Training and evaluating the Applm model
With the pre-computed embeddings, the Applm model can be easily trained and evaluated using the `run_train.py` file in `code/4_Training_Applm_RF_Model`. The `run_train.py` file reads the `train.fa` and `test.fa` in the directory and then trains the model using the precomputed `embed` files.
```
# Executing in the `code/4_Training_Applm_RF_Model` directory

python run_train.py \
    --training_directory 1_train_splits_fa/ex1 \ # Directory containing train.fa and test.fa
    --embedding_directory 0_embeddings/avgpool_ohe \
    --embed ohe \
    --output_directory 2_results/ex1
```
The reults will be saved to `<output_directory>/<embed>/test_labeled.csv` with the prediction score for each instance in the test set. Optionally, you may run `code/4_Training_Applm_RF_Model/notebook/2_compute_metrics.ipynb` on the `test_labeled.csv` to compute the AUROC and AUPRC using the `PRROC` package.

## System Requirements

- **Operating System:** Ubuntu 22.04

## Dependencies

`environment.yaml`
-   For extracting embeddings (except xTrimoPGLM), the similarity-aware pipeline, and training and testing Applm.
-   Main dependencies include
    -   python                    3.10.9 
    -   numpy                     1.26.4
    -   torch                     2.4.1
    -   transformers              4.39.2
    -   fair-esm                  2.0.0

`environment_xtrimopglm.yaml`
-   For extracting xTrimoPGLM embeddings.
-   Main dependencies include
    -   python                    3.13.2 
    -   numpy                     2.2.4
    -   torch                     2.6.0
    -   transformers              4.51.1
    -   deepspeed                 0.16.5

## Example Data

Internal and external splits used in this study can be found in the directories `dataset/Internal` and `dataset/External` respectively. The splits are saved in `fasta` format as well as `csv` format.  

In `dataset/Internal`, *Hard Balanced* splits at T<sub>s</sub> ∈ (1.0 ,0.5, 0.4, 0.3), and T<sub>c</sub> ∈ (0.0, 0.4, 0.5, 0.6, 0.7) are included. 

In `dataset/External`, *Hard Balanced* training sets for all 6 external test sets at T<sub>s</sub> ∈ (1.0 ,0.5), and T<sub>c</sub> ∈ (0.0, 0.4, 0.6) are included. 
## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## How to Cite

```
TBD
```

## Authors and Acknowledgements

Brian Shing-Hei Wong<sup>1</sup>, Joshua Mincheol Kim<sup>1</sup>, Sin-Hang Fung<sup>1,2</sup>, Qing Xiong<sup>3</sup>, Kelvin Fu-Kiu Ao<sup>1</sup>, Junkang Wei<sup>1,4</sup>, Ran Wang<sup>1</sup>, Dan Michelle Wang<sup>1</sup>, Jingying Zhou<sup>1</sup>, Bo Feng<sup>1</sup>, Alfred Sze-Lok Cheng<sup>1</sup>, Kevin Y. Yip<sup>5,6,7,8,</sup>\*, Stephen Kwok-Wing Tsui<sup>1,9,</sup>\*, Qin Cao<sup>1,2,9,</sup>*

1. School of Biomedical Sciences, The Chinese University of Hong Kong, Shatin, New Territories, Hong Kong SAR, China
2. Shenzhen Research Institute, The Chinese University of Hong Kong, Shenzhen, China
3. Department of Health Technology and Informatics, The Hong Kong Polytechnic University, Hung Hom, Kowloon, Hong Kong SAR, China
4. Department of Computational Medicine and Bioinformatics, University of Michigan, Ann Arbor, MI, USA
5. Center for Data Sciences, Sanford Burnham Prebys Medical Discovery Institute, La Jolla, CA, USA
6. Cancer Genome and Epigenetics Program, NCI-Designated Cancer Center, Sanford Burnham Prebys Medical Discovery Institute, La Jolla, CA, USA
7. Center for Neurologic Diseases, Sanford Burnham Prebys Medical Discovery Institute, La Jolla, CA, USA
8. Department of Computer Science and Engineering, The Chinese University of Hong Kong, Shatin, New Territories, Hong Kong SAR, China
9. Hong Kong Bioinformatics Centre, The Chinese University of Hong Kong, Shatin, New Territories, Hong Kong SAR, China
---
*To whom correspondence should be addressed. Emails: kyip@sbpdiscovery.org (K.Y. Yip), kwtsui@cuhk.edu.hk (S.K.W. Tsui), qcao@cuhk.edu.hk (Q. Cao)

## Contact Information

```
TBD
```
