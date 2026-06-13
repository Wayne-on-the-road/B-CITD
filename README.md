# Graph Intelligence Enhanced Bi-Channel Insider Threat Detection

This repository contains the experimental code and processed data used for the paper:

**Graph Intelligence Enhanced Bi-Channel Insider Threat Detection**  
Wei Hong, Jiao Yin, Mingshan You, Hua Wang, Jinli Cao, Jianxin Li, Ming Liu  
In *International Conference on Network and System Security*, pp. 86–102  
Springer Nature Switzerland, 2022

## Overview

Insider threat detection is a challenging cybersecurity task because malicious behaviours are often rare, subtle, and embedded within large volumes of normal user activity. This project implements the experimental pipeline for a graph-intelligence-enhanced bi-channel insider threat detection framework.

The proposed approach combines user-level behavioural representations with graph-based relational intelligence. It is designed to capture both individual behavioural patterns and inter-user relationships, thereby improving the detection of insider threats in enterprise activity logs.

## Repository Structure

```text
.
├── CERT4.2/
│   └── Processed user-day graph data and preprocessing-related files
├── GCN_mean-upload.py
├── SVM-upload.py
├── cnn-trial-cert-whole-mean-upload.py
├── data_process_new_final-upload.py
├── feature_extract_multi_final-upload.py
├── generate_train_test_data_upload.py
└── README.md
```

## Dataset

The experiments are based on the **CERT 4.2 insider threat dataset**.

Due to the large size of the original CERT dataset, the raw files are not included in this repository. The `CERT4.2` folder contains the processed user-day graph dataset constructed for the experiments, along with selected example files related to preprocessing.

Before running the full preprocessing pipeline, users need to download the following raw CERT files from the CMU CERT research center:

```text
device.csv
logon.csv
```

These files should be placed in the appropriate directory expected by the preprocessing scripts.

## Main Scripts

### `GCN_mean-upload.py`

Runs the graph convolutional network component and evaluates the performance of the graph-based feature extractor. This script is used to examine the contribution of graph intelligence and inter-user relational features.

### `cnn-trial-cert-whole-mean-upload.py`

Runs the bi-channel detection experiment. This script is used to compare the performance of the inner-user behavioural channel and the proposed bi-channel framework.

### `SVM-upload.py`

Runs the SVM-based comparison experiment. This script provides a traditional machine learning baseline for evaluating the effectiveness of the proposed representation and feature construction strategy.

### `feature_extract_multi_final-upload.py`

Extracts behavioural features from the CERT activity logs. This script is part of the preprocessing pipeline and should be executed before model training when reconstructing the dataset from raw files.

### `data_process_new_final-upload.py`

Processes and reorganises the extracted features into the required experimental format.

### `generate_train_test_data_upload.py`

Generates training and testing datasets for the detection experiments.

## Suggested Execution Order

If you want to reproduce the preprocessing workflow from raw CERT files, run the following scripts in order:

```bash
python feature_extract_multi_final-upload.py
python data_process_new_final-upload.py
python generate_train_test_data_upload.py
```

After preprocessing, run the model and comparison experiments:

```bash
python GCN_mean-upload.py
python cnn-trial-cert-whole-mean-upload.py
python SVM-upload.py
```

Please ensure that all required CSV files are placed in the correct directories before running the scripts.

## Reproducibility Notes

The original CERT 4.2 dataset is large and cannot be directly hosted in this repository. Therefore, this repository provides the processed graph-based user-day dataset used in the experiments, together with the scripts required to understand and reproduce the main preprocessing and modelling procedures.

Because file paths and local environments may differ, users may need to adjust directory settings in the scripts before execution.

## Citation

If you use this repository or refer to the method in your research, please cite the following paper:

```bibtex
@inproceedings{hong2022graph,
  title     = {Graph Intelligence Enhanced Bi-Channel Insider Threat Detection},
  author    = {Hong, Wei and Yin, Jiao and You, Mingshan and Wang, Hua and Cao, Jinli and Li, Jianxin and Liu, Ming},
  booktitle = {International Conference on Network and System Security},
  pages     = {86--102},
  year      = {2022},
  publisher = {Springer Nature Switzerland}
}
```

## Keywords

Insider threat detection; graph intelligence; graph neural networks; bi-channel detection; behavioural analytics; cybersecurity; CERT dataset.

## Contact

For questions about the code or experiments, please contact the corresponding author or open an issue in this repository.
