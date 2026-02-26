# 🧬 Bio-Clustering Benchmarks

This repository contains a robust, modular pipeline for preprocessing and evaluating various clustering algorithms on real-world biological and medical datasets. 

It was built to serve as a standardized benchmarking environment to compare traditional baseline clustering algorithms against **RLAC** and **MDH**, two specialized continuous-optimization clustering models implemented as part of my academic thesis.

## 📊 Datasets Evaluated
The pipeline is currently configured to process and cluster three datasets:
1. **Breast Cancer Wisconsin (Diagnostic):** Binary clustering (Malignant vs. Benign).
2. **Vertebral Column:** Multi-class orthopedic clustering.
3. **Anuran Calls (MFCCs):** A complex acoustic dataset of frog calls. This dataset evaluates the models across hierarchical taxonomic targets (`Family`, `Genus`, `Species`).

## 🤖 Models Tested
**Baseline Algorithms:**
* K-Means
* Spectral Clustering (N-Cut)
* Agglomerative Clustering (Single Linkage)

**Custom/Thesis Models:**
* **[RLAC](#)** (Robust Locality Agglomerative Clustering) - *Grid search applied across various density projection methods.*
* **[MDH](#)** (Minimum Density Hyperplanes) - *Continuous optimization hyperplane clustering.*

*(Note: The source code for RLAC and MDH are maintained in their own standalone repositories, linked above).*

## ⚙️ Pipeline Features
Instead of procedural notebook scripts, this repository uses a strictly modular software engineering architecture:
* **Automated Preprocessing (`preprocessing.py`):** Handles feature de-duplication, IQR-based outlier management (both row removal and winsorization/clipping strategies), and standard scaling.
* **Universal Data Loaders (`data_loader.py`):** Easily expandable functions to load and isolate features/targets without cluttering the main logic.
* **Standardized Evaluation (`experiments.py`):** Automatically evaluates all predictions against ground-truth labels using **AMI** (Adjusted Mutual Information), **ARI** (Adjusted Rand Index), and **Silhouette Scores**.

## 📁 Repository Structure
```text
bio-clustering-benchmarks/
│
├── data/                    # Dataset CSV files (Breast Cancer, Vertebral, Anuran)
├── custom_models/           # Clone/Place rlac.py and mdh.py here
│
├── src/                     
│   ├── data_loader.py       # Functions to parse specific CSV structures
│   ├── preprocessing.py     # Outlier handling, scaling, and deduplication
│   └── experiments.py       # Model initialization and metric scoring
│
└── main.py                  # Master execution file
```

## 🚀 How to Run
Clone the repository
```Bash
git clone https://github.com/geosh0/bio-clustering-benchmarks.git
cd bio-clustering-benchmarks
```
## Install dependencies:

```Bash
pip install pandas numpy scikit-learn
```
## Run the Master Pipeline:
```Bash
python main.py
```
* The script will process all datasets, test the outlier strategies, run the grid searches, and output a final, aggregated Pandas DataFrame showing the top performing configurations sorted by AMI.

## Bibliography
* [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* [Vertebral Column Data Set](https://www.kaggle.com/datasets/caesarlupum/vertebralcolumndataset)
* [Anuran Calls (MFCCs)](https://www.kaggle.com/datasets/yasserhessein/anuran-calls-mfccs)
