# Academic-Success-Classifier
====================

Machine Learning project to classify students via predicting their academic success. Predicts if a student is likely to graduate or dropout given specific metrics.

## Contents
--------
- What this project does
- Prerequisites
- Installation (Poetry or pip/venv)
- Configure Kaggle API (.env)
- Quickstart: download data, preprocess, train, and validate
- Outputs and artifacts
- Troubleshooting
- FAQ
- Contributing
- License


## Before you start (read me if you’re new!)
----------------------------------------
If you’re not comfortable with Python tools yet, follow these tips first:

- You need Python 3.11–3.14 installed.
  - Check your version (Windows PowerShell):
    ```powershell
    python --version
    ```
  - Check your version (macOS/Linux):
    ```bash
    python3 --version
    ```

- You need Git to clone this repository.
  - Check Git:
    ```powershell
    git --version
    ```
    ```bash
    git --version
    ```

- How to open a terminal:
  - Windows: Press Win key, type “PowerShell”, press Enter.
  - macOS: Open “Terminal” app.
  - Linux: Open your distribution’s terminal app.

- Where to run commands: First navigate into the folder where you want the project. Example:
  - Windows (PowerShell):
    ```powershell
    cd $HOME\Documents
    ```
  - macOS/Linux:
    ```bash
    cd "$HOME/Documents"
    ```

- Copy exactly what is inside each code box as a single line. Every command below is split so you can copy one line at a time.


## What this project does
----------------------
At a high level:
1. Downloads the “Stanford Car Dataset by classes folder” from Kaggle.
2. Filters/reorganizes the dataset into `./data/processed_csv/{train/test}.csv`.
3. Builds various models with a custom classifier head; finds the most accurate model.
4. Saves the trained model to `./model/student_success_model.pkl` and evaluates on the test set.



## Prerequisites
-------------
Software
- Python 3.11–3.14 (project targets >=3.11, <3.15).
- Git (to clone the repository).
- One of:
  - Poetry (recommended), or
  - pip + venv.
- Kaggle account and API credentials (see “Configure Kaggle API”).
- Kaggle CLI available in your shell. Installing the `kaggle` Python package provides the `kaggle` command but you may need to re‑open your shell so PATH updates take effect.

Hardware
- CPU‑only works (***extremely slow***); GPU (NVIDIA) recommended for performance. Model was trained on an Nvidia L40S -> ~1hrs on GPU.
- If using GPU, install an appropriate NVIDIA driver. Prebuilt PyTorch usually bundle CUDA runtime.

Disk and bandwidth
- Dataset download, repository files, extraction, and other files require ~4GB gigabytes free. ***Ensure you have 4GB of space free on your disk before cloning.***


## Installation
------------

### Python pip w/ Poetry
1. Clone the repository.
   - Windows (PowerShell):
     ```powershell
     git clone https://github.com/Admeen3581/AutomotiveClassifier.git
     cd AutomotiveClassifier
     ```
   - macOS/Linux (Bash):
     ```bash
     git clone https://github.com/Admeen3581/AutomotiveClassifier.git
     cd AutomotiveClassifier
     ```
2. Install Poetry:
   - Install/Upgrade pip first:
     ```bash
     python -m pip install --upgrade pip
     ```
   - Install Poetry:
     ```bash
     pip install poetry
     ```
3. Install project dependencies from pyproject using Poetry:
   - ```bash
     poetry install
     ```
4. Build your .env file in the *root* directory:
   - ```env
     KAGGLE_API_USER=<your_kaggle_username>
     ```
   - ```env
     KAGGLE_API_TOKEN=<your_kaggle_api_key>
     ```
5. Optional: Use Poetry to run the app:
   - ```bash
     poetry run python main.py
     ```

## Configure Kaggle API (.env)
---------------------------
The dataset is downloaded via the Kaggle CLI and requires API credentials.

1. Create a Kaggle account (https://www.kaggle.com/).
2. Generate an API token: Account settings → Create New API Token. This downloads `kaggle.json` containing `username` and `key`.
3. In this project’s root directory, create a file named `.env` with these two lines (each line is separate):
   - ```env
     KAGGLE_API_USER=<your_kaggle_username>
     ```
   - ```env
     KAGGLE_API_TOKEN=<your_kaggle_api_key>
     ```
4. Ensure the `kaggle` command is available in your shell. If you just installed it, close and reopen your terminal.
5. You must accept the dataset’s terms on Kaggle and join the competition to download it. Visit the dataset page and click “I Understand and Accept” if prompted:


## Quickstart
----------
Run the end‑to‑end pipeline. On first run, this will:
- Create `./data/` if it does not exist.
- Download and extract the Kaggle dataset into `./data/processed_csv/`.

Using Poetry:
```bash
poetry run python main.py
```

Using pip/venv:
```bash
python main.py
```

Notes
- If `./data` already exists, dataset initialization will print a message and skip downloading. Delete `./data` to force a fresh re‑init.


## Outputs and artifacts
---------------------
- `./model/student_success_model.pkl` — Saved model weights after training.
- `./model/scaler.pkl` — Saved scaler weights after training.


## Troubleshooting
---------------
- “kaggle: command not found” or Kaggle CLI not recognized
  - Ensure the environment has Kaggle installed (it is declared in pyproject and installed by Poetry). Re‑open your terminal so the `kaggle` command is on PATH.
  - Confirm you accepted the dataset terms on Kaggle.
  - Ensure `.env` contains `KAGGLE_API_USER` and `KAGGLE_API_TOKEN`.

- Dataset init keeps skipping with a message that `./data` exists
  - Delete the `./data` directory to force a fresh init. The initializer returns early if it detects an existing data directory.
 
- `FileNotFoundError: Could not find CSV file at ./data/anno_test_filtered.csv. Possible a dataset API call issue.`
  - - Delete the `./data` directory to force a fresh init.

- CUDA/CuDNN errors
  - If you don’t need GPU acceleration, run on CPU (it will auto‑detect). If you do, ensure your NVIDIA driver is up to date and that your Torch install matches your CUDA runtime. Installing the default pip/Poetry wheel typically includes a compatible CUDA runtime.

- OpenCV cannot read images (`Image not found`)
  - Ensure the dataset successfully downloaded and reorganized and that your working directory is the project root when running `main.py`.

- `TypeError: expected str, bytes or os.PathLike object, not NoneType`
  - Ensure you .env file is setup correctly.


## FAQ
---
- How do I re‑run preprocessing?
  - Delete the `./data` directory and run `main.py` again.


## Contributing
------------
Pull requests are welcome. Please keep to the existing code style and structure. If adding dependencies, update `pyproject.toml` and ensure installation works with both Poetry and pip/venv. Consider adding or updating tests under `test/` as appropriate.


## License
-------
This project is licensed under the MIT License. See `LICENSE.md` for details.
