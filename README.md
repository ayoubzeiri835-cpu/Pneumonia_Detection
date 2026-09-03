# Pneumonia Detection with Explainable AI

> Deep learning pipeline for pneumonia detection from chest X-ray images, enhanced with Grad-CAM visual explanations and an OWL ontology.

## Overview

This project explores automated **pneumonia detection from chest X-ray images** using transfer learning with **EfficientNetB0**.

The original workflow was developed in a Kaggle notebook and combines:

- Transfer learning with EfficientNetB0 pretrained on ImageNet
- Image augmentation and EfficientNet preprocessing
- Focal Loss to focus learning on difficult examples
- Accuracy, Precision, Recall, AUC and F1 monitoring
- Validation-based early stopping and learning-rate reduction
- Test-set threshold optimization using a weighted recall objective
- Grad-CAM visual explanations
- Left/right/bilateral activation-zone analysis
- OWL ontology with Owlready2 for knowledge-based explanations
- A Gradio interface for interactive inference

The project is intended as an **AI/XAI research and portfolio project**, not as a clinical diagnostic system.

## Project Pipeline

```text
Chest X-ray
     │
     ▼
Image preprocessing
     │
     ▼
Data augmentation (training)
     │
     ▼
EfficientNetB0 (ImageNet)
     │
     ▼
Binary classifier
     │
     ▼
Pneumonia / Normal
     │
     ├──────────────► Confidence
     │
     ▼
Grad-CAM
     │
     ▼
Activation heatmap
     │
     ▼
Dominant activation zone
     │
     ▼
OWL ontology
     │
     ▼
Human-readable explanation
     │
     ▼
Gradio interface
```

## Model

```text
Input (224 × 224 × 3)
        ↓
EfficientNetB0 (ImageNet weights, frozen)
        ↓
GlobalAveragePooling2D
        ↓
Dense(128, ReLU)
        ↓
BatchNormalization
        ↓
Dropout(0.5)
        ↓
Dense(1, Sigmoid)
```

Training configuration:

- Image size: `224 × 224`
- Batch size: `32`
- Optimizer: Adam
- Learning rate: `2e-4`
- Epochs: up to `30`
- Early stopping: validation loss, patience `7`
- ReduceLROnPlateau: factor `0.3`, patience `3`
- Focal Loss: `gamma=2`, `alpha=0.6`
- Augmentation: rotation, zoom and contrast

## Threshold Optimization

The training workflow evaluates thresholds from `0.30` to `0.89` and selects the threshold maximizing:

```text
0.4 × Recall(NORMAL) + 0.6 × Recall(PNEUMONIA)
```

The selected threshold is saved to:

```text
results/threshold.json
```

The Gradio application reuses this threshold when the file is available.

## Explainable AI

### Grad-CAM

Grad-CAM generates a heatmap representing image regions that contributed strongly to the model output.

The application displays:

1. Original chest X-ray
2. Grad-CAM heatmap
3. Heatmap overlaid on the X-ray

### Activation-zone analysis

The heatmap is split into left and right halves. The activation intensity is compared and the system reports:

- `gauche`
- `droite`
- `bilatérale`

This is an exploratory visualization heuristic, **not anatomical localization or medical segmentation**.

## Ontology

The project uses **Owlready2** to define a lightweight knowledge model:

```text
Organe
 └── Poumon
      ├── PoumonDroit
      └── PoumonGauche

EtatPulmonaire
 ├── Normal
 └── Pneumonie

SigneRadiologique
 ├── Opacite
 └── Infiltration
```

Relationships include:

```text
a_localisation
a_signe
indique
```

The current knowledge model contains rules such as:

```text
Opacite ── indique ──► Pneumonie
Infiltration ── indique ──► Pneumonie
```

The XAI pipeline combines the model prediction, Grad-CAM zone and ontology concepts to generate a human-readable explanation.

## Dataset

The notebook uses the **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney, accessed through Kaggle.

Expected structure:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The training directory is split into training and validation subsets using an `80/20` split.

**Dataset files are intentionally not included in this repository.**

## Repository Structure

```text
pneumonia-detection-xai/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── pneumonia.ipynb
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── xai.py
│   └── ontology.py
│
├── app/
│   ├── __init__.py
│   └── app.py
│
├── ontology/
│   └── pneumo_ontology.owl
│
├── models/
│   └── README.md
│
└── results/
    ├── README.md
    └── .gitkeep
```

## Installation

Python 3.10+ is recommended.

```bash
git clone <YOUR_REPOSITORY_URL>
cd pneumonia-detection-xai

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Configuration

Set `PNEUMONIA_DATASET_DIR` to the directory containing `train/` and `test/`.

Windows PowerShell:

```powershell
$env:PNEUMONIA_DATASET_DIR="C:\path\to\chest_xray"
```

Linux/macOS:

```bash
export PNEUMONIA_DATASET_DIR="/path/to/chest_xray"
```

## Training

```bash
python -m src.train
```

The script creates:

```text
models/best_model_tl.keras
results/confusion_matrix.png
results/loss_curve.png
results/threshold.json
```

## Run the XAI Demo

After training:

```bash
python app/app.py
```

The Gradio interface provides:

- predicted class
- confidence
- original image
- Grad-CAM visualization
- dominant activation zone
- ontology-based explanation

## Results

Add your final reproduced evaluation values here:

| Metric | Value |
|---|---:|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-score | — |
| AUC | — |

The notebook generates the classification report and confusion matrix. Replace the placeholders above with the values from your final experiment.

## Limitations

- The model is trained on a specific chest X-ray dataset and may not generalize to other populations, devices or clinical settings.
- Grad-CAM is a model-attribution visualization; it does not prove that a highlighted region is pathological.
- The left/right/bilateral detector is a simple heatmap heuristic, not precise anatomical localization.
- The ontology is a small project-specific knowledge representation, not a complete medical ontology.
- The system is not clinically validated.

## Future Improvements

- Fine-tune selected EfficientNetB0 layers after frozen-base training
- Add probability calibration and more robust threshold selection
- Compare additional XAI methods
- Improve anatomical localization
- Expand the ontology and inference rules
- Add experiment tracking
- Add unit tests and CI
- Containerize the Gradio application
- Add a reproducible inference API

## Technologies

`Python` · `TensorFlow/Keras` · `EfficientNetB0` · `OpenCV` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Owlready2` · `Gradio`

## Disclaimer

This project is for educational, research and portfolio purposes. It must not be used to make medical diagnoses or clinical decisions.
