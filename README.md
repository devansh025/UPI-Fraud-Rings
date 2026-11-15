# UPI Fraud Rings Detection

A multiclass machine-learning project for identifying potentially
fraudulent UPI transaction patterns using transaction, account,
merchant, device, temporal, and graph-derived features. The model in the
current notebook is **CatBoostClassifier**.

> **Project status:** The notebook trains a CatBoost model and generates
> `submission_catboost.csv`. The notebook does not currently report a
> final macro-F1 score or establish production readiness.

## Problem

The task is to classify each UPI transaction into one of the labels
provided in the training dataset. The training data contains a `label`
column; the test data is used for generating predictions.

The notebook is structured around: 1. Loading and inspecting the
supplied dataset. 2. Exploring class distribution, time distribution,
and feature drift. 3. Joining account and merchant metadata. 4.
Engineering transaction-level, temporal, velocity, and graph features.
5. Creating a chronological training/validation split. 6. Training a
multiclass CatBoost model. 7. Predicting test labels and exporting a
submission CSV.

## Dataset

The notebook expects the following files:

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `train.csv`                         Labeled transactions used for
                                      training and validation

  `test.csv`                          Unlabeled transactions for
                                      prediction

  `accounts.csv`                      Account metadata joined using
                                      account identifiers

  `merchants.csv`                     Merchant metadata joined using
                                      merchant identifiers
  -----------------------------------------------------------------------

The original archive used by the notebook is named
`upi-fraud-rings-gradient-wars.zip`. These data files are not included
in this repository by default. Obtain them from the dataset/competition
source and place them in the expected directory.

### Expected local path

The current notebook contains a Windows-specific path:

``` python
ZIP_PATH = r"C:\Users\Lenovo\Downloads\upi-fraud-rings-gradient-wars.zip"
DATA_DIR = r"C:\Users\Lenovo\Downloads\upi-fraud-rings"
```

Change these values to match your machine. The notebook extracts the
archive when the expected CSV files are not already present in
`DATA_DIR`.

## Requirements

-   Python 3
-   Jupyter Notebook support in VS Code or JupyterLab
-   pandas
-   NumPy
-   Matplotlib
-   scikit-learn
-   CatBoost

Install the dependencies from a terminal:

``` bash
python -m pip install pandas numpy matplotlib scikit-learn catboost jupyter
```

The notebook also contains a CatBoost installation cell:

``` python
!pip -q install catboost
```

## Run the notebook

1.  Clone or download this project.
2.  Obtain the dataset archive and extract it, or update `ZIP_PATH` and
    `DATA_DIR` in the notebook.
3.  Open the `.ipynb` file in VS Code.
4.  Select a Python kernel with the required packages installed.
5.  Run the notebook cells in order, from data loading through feature
    engineering and model training.
6.  After the CatBoost training cell completes, run the final prediction
    cell.

The model is trained in the CatBoost cell using the chronological split
prepared earlier in the notebook. The prediction cell uses the in-memory
fitted `model`; it does not train the model again. Keep the notebook
kernel alive between training and prediction.

## Feature engineering

The notebook creates several groups of features.

### Transaction and time features

-   Transaction hour, day, weekday, day of year
-   Weekend and night-time indicators
-   Log-transformed transaction amount
-   Amount-to-balance ratio
-   Remaining balance

### Account, merchant, device, and location features

-   Account and merchant metadata joined to transactions
-   Log-transformed SIM, device, and account age
-   Account and merchant age at transaction time
-   Missing-metadata indicators
-   Sender-home-city and merchant/device-city consistency indicators
-   Categorical identifiers and attributes, including sender, receiver,
    and device IDs

### Historical velocity features

Past-only rolling features are calculated for sender, receiver, and
device activity, including:

-   Sender transaction counts over 5 minutes, 1 hour, and 24 hours
-   Sender amount sums over 1 hour and 24 hours
-   Receiver transaction counts over 1 hour and 24 hours
-   Device transaction counts over 1 hour and 24 hours

The rolling-feature function excludes transactions at the same timestamp
as the current transaction.

### Graph-derived features

The notebook constructs an unlabeled transaction graph from train and
test transactions and calculates:

-   Number of distinct typed receivers associated with a sender
-   Number of distinct senders associated with a typed receiver
-   Number of distinct sender accounts associated with a device
-   Log-transformed versions of these graph counts

## Validation strategy

The notebook uses a chronological split with a cutoff of **2026-09-15**:

-   Training: transactions before the cutoff
-   Validation: transactions on or after the cutoff

This is intended to better reflect a time-based test set than a random
split. The notebook also explores monthly distribution and train/test
feature drift.

**Validation caveat:** The graph-derived degree features are calculated
from the combined unlabeled train and test transaction graph. These
global graph counts can include information from transactions later than
a training or validation transaction. Therefore, the chronological split
alone does not make the full feature pipeline strictly time-safe. For a
rigorous time-based evaluation, compute graph features using only
information available before each transaction (or otherwise justify
transductive graph features under the competition rules) and regenerate
the validation features before interpreting validation performance.

## Model

The current model is `CatBoostClassifier` configured for multiclass
classification:

``` python
CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.08,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    auto_class_weights="SqrtBalanced",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=150,
    allow_writing_files=False
)
```

Categorical columns are identified from the training feature data and
missing categorical values are filled with `"__MISSING__"` before
fitting.

The current configuration does **not** specify `task_type="GPU"`, so
CatBoost uses its default CPU training mode. To experiment with GPU
training on a compatible NVIDIA/CUDA setup, configure CatBoost for GPU
and verify compatibility; GPU training requires a new training run.

## Output

The final notebook cell writes:

``` text
submission_catboost.csv
```

The CSV contains exactly two columns:

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `txn_id` or `id`                    Identifier copied from the test
                                      dataset, depending on which column
                                      exists

  `label`                             Predicted class from the trained
                                      CatBoost model
  -----------------------------------------------------------------------

The number of output rows should match the number of rows in `test.csv`.

## Important implementation notes

-   Run the notebook cells in order so that the feature tables and
    fitted model exist in memory.
-   The final prediction cell expects `test_fe` to have the same feature
    columns, ordering, and categorical preprocessing used for training.
-   If the notebook kernel is restarted after training, the in-memory
    model is lost. Rerun training or explicitly save and load the fitted
    CatBoost model.
-   The notebook currently saves the submission in the notebook's
    current working directory.
-   The notebook contains exploratory cells and some repeated
    calculations. These can be cleaned up separately, but avoid deleting
    cells that create variables used later.
-   A successful CSV export confirms that predictions were written; it
    does not by itself establish model quality. Evaluate class-wise
    precision/recall and macro-F1 on a leakage-safe validation set.

## Repository structure

A suggested layout is:

``` text
upi-fraud-rings/
├── README.md
├── starter_notebook.ipynb
└── data/                  # Keep dataset files local; usually do not commit them
    ├── train.csv
    ├── test.csv
    ├── accounts.csv
    └── merchants.csv
```

## Disclaimer

This is a machine-learning project/notebook for experimentation and
competition-style prediction. It is not a deployed fraud-prevention
system and should not be used to make real financial decisions without
extensive independent validation, monitoring, security review, and
appropriate safeguards.
