"""
data_loader.py
--------------
Load the ODIR-5K dataset CSV file, validate image paths,
and return image paths with multi-label targets.
"""

import os
import numpy as np
import pandas as pd


# Disease label columns in the CSV
DISEASE_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

DISEASE_NAMES = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration',
    'H': 'Hypertension',
    'M': 'Myopia',
    'O': 'Other Abnormalities',
}


def load_dataset(csv_path: str, image_dir: str):
    """
    Load the ODIR-5K dataset.

    Parameters
    ----------
    csv_path : str
        Path to ``full_df.csv``.
    image_dir : str
        Directory containing the retinal fundus images.

    Returns
    -------
    image_paths : np.ndarray
        Array of valid image file paths.
    labels : np.ndarray
        Multi-label target array of shape (N, 8).
    df : pd.DataFrame
        The filtered dataframe for further inspection.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    if 'filename' not in df.columns:
        raise ValueError("CSV must contain a 'filename' column.")
    for col in DISEASE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"CSV missing disease column: {col}")

    # Build full paths and filter out missing images
    df['image_path'] = df['filename'].apply(
        lambda fn: os.path.join(image_dir, fn)
    )
    mask = df['image_path'].apply(os.path.isfile)
    missing = (~mask).sum()
    if missing > 0:
        print(f"[data_loader] Warning: {missing} images not found on disk — skipped.")
    df = df[mask].reset_index(drop=True)

    image_paths = df['image_path'].values
    labels = df[DISEASE_COLUMNS].values.astype(np.float32)

    print(f"[data_loader] Loaded {len(image_paths)} samples with {labels.shape[1]} labels.")
    return image_paths, labels, df


if __name__ == "__main__":
    # Quick test
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = os.path.join(BASE, "dataset", "odir", "full_df.csv")
    imgs = os.path.join(BASE, "dataset", "odir", "preprocessed_images")
    paths, lbls, dataframe = load_dataset(csv, imgs)
    print(dataframe.head())
