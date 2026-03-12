# 👁️ RetinAI — Multi-Label Retinal Disease Detection

Deep learning system for detecting co-existing eye diseases from retinal fundus images using the **ODIR-5K** dataset and **DenseNet121** transfer learning.

## Detected Diseases

| Code | Disease |
|------|---------|
| N | Normal |
| D | Diabetes |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration |
| H | Hypertension |
| M | Myopia |
| O | Other Abnormalities |

## Project Structure

```
EDI_FINAL/
├── dataset/odir/
│   ├── preprocessed_images/   # retinal fundus images
│   └── full_df.csv           # multi-label CSV
├── src/
│   ├── data_loader.py        # load CSV + validate images
│   ├── preprocessing.py      # resize, normalise, CLAHE
│   ├── dataset_builder.py    # tf.data pipeline + augmentation
│   ├── model.py              # DenseNet121 architecture
│   ├── train.py              # two-phase training
│   ├── evaluate.py           # metrics + reports
│   └── predict.py            # single-image prediction
├── models/
│   └── retinal_model.keras   # saved model (after training)
├── app/
│   ├── main.py               # FastAPI server
│   ├── app.py                # app factory
│   ├── templates/index.html  # web UI
│   └── static/style.css      # styling
├── reports/                  # evaluation outputs
├── notebooks/                # exploration
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Place `full_df.csv` inside `dataset/odir/`. Retinal images should be in `dataset/odir/preprocessed_images/`.

### 3. Train the model

```bash
python src/train.py
```

Training runs in two phases:
- **Phase 1** — Frozen DenseNet121 backbone, train classifier head (10 epochs)
- **Phase 2** — Unfreeze top layers, fine-tune (20 epochs)

The best model is saved to `models/retinal_model.keras`.

### 4. Evaluate

```bash
python src/evaluate.py
```

Results are saved to `reports/`.

### 5. Predict on a single image

```bash
python -m src.predict path/to/image.jpg
```

### 6. Run the API

```bash
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to use the web interface.

## Tech Stack

- **TensorFlow / Keras** — model training & inference
- **DenseNet121** — ImageNet pre-trained backbone
- **FastAPI** — REST API server
- **scikit-learn** — evaluation metrics
- **OpenCV** — image preprocessing

## License

This project is for educational and research purposes.
