# Brain Tumour and Pneumonia Classification

Python/Keras project for classifying medical images with convolutional neural networks.

The project is now organized under one main directory: `BrainTumoour`.

## Project Structure

```text
.
|-- BrainTumoour/
|   |-- train_brain_tumor.py      # Train the brain tumour classifier
|   |-- train_pneumonia.py        # Train the pneumonia classifier
|   |-- predict_brain_tumor.py    # Run a sample brain tumour prediction
|   |-- gui.py                    # Tkinter GUI for brain tumour prediction
|   |-- sample_images/            # Small demo images
|   `-- data/                     # Local datasets, ignored by Git
|       |-- brain_tumor_training/
|       |-- brain_tumor_testing/
|       |-- pneumonia_train/
|       `-- pneumonia_test/
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Classes

Brain tumour MRI classification:

- `glioma_tumor`
- `meningioma_tumor`
- `no_tumor`
- `pituitary_tumor`

Chest X-ray pneumonia classification:

- `NORMAL`
- `PNEUMONIA`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Dataset Layout

The scripts expect datasets inside `BrainTumoour/data`:

```text
BrainTumoour/data/
|-- brain_tumor_training/
|   |-- glioma_tumor/
|   |-- meningioma_tumor/
|   |-- no_tumor/
|   `-- pituitary_tumor/
|-- brain_tumor_testing/
|   |-- glioma_tumor/
|   |-- meningioma_tumor/
|   |-- no_tumor/
|   `-- pituitary_tumor/
|-- pneumonia_train/
|   |-- NORMAL/
|   `-- PNEUMONIA/
`-- pneumonia_test/
    |-- NORMAL/
    `-- PNEUMONIA/
```

`BrainTumoour/data` is ignored by Git because medical image datasets are usually large. Use Git LFS or a dataset download link if you need to share the full data.

## Usage

Train the brain tumour model:

```bash
python BrainTumoour/train_brain_tumor.py
```

This creates:

```text
BrainTumoour/brain_tumor_model.h5
```

Train the pneumonia model:

```bash
python BrainTumoour/train_pneumonia.py
```

This creates:

```text
BrainTumoour/pneumonia_model.h5
```

Run the sample brain tumour prediction:

```bash
python BrainTumoour/predict_brain_tumor.py
```

Launch the GUI:

```bash
python BrainTumoour/gui.py
```

Before using prediction or the GUI, train the brain tumour model so `BrainTumoour/brain_tumor_model.h5` exists.

## Cleanup Notes

- Generated Python cache files were removed.
- Windows `Zone.Identifier` download metadata files were removed.
- The old nested project folder was replaced with the cleaner `BrainTumoour` directory.
- The GUI no longer depends on machine-specific external image assets.

## Disclaimer

This project is for learning and experimentation only. It is not intended for clinical diagnosis or medical decision-making.
