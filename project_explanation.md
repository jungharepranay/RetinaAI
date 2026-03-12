# 🔬 Retinal Disease Detection — Complete Viva Preparation Guide

> **Project**: Multi-Label Retinal Disease Detection using Deep Learning & ODIR-5K  
> **Architecture**: DenseNet121 with Transfer Learning  
> **Dataset**: ODIR-5K (Ocular Disease Intelligent Recognition)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Machine Learning Pipeline](#2-machine-learning-pipeline)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
4. [The Notebook: Experiments & Visualizations](#4-the-notebook-experiments--visualizations)
5. [Understanding the Results](#5-understanding-the-results)
6. [Viva Questions & Answers](#6-viva-questions--answers)
7. [Real-World Analogies](#7-real-world-analogies)
8. [5–7 Minute Presentation Script](#8-57-minute-presentation-script)

---

## 1. Project Overview

### What Does This Project Do?

This project is an **AI-powered diagnostic assistant** that looks at retinal fundus images (photographs of the back of the eye) and detects which diseases are present. Think of it as a "second opinion" tool that could help ophthalmologists screen patients faster.

A **fundus camera** captures a photograph of the retina — the light-sensitive tissue at the back of the eye. By examining patterns like blood vessel changes, spots, or discolouration in these images, our deep learning model can predict the presence of **eight different conditions**:

| Code | Disease | What It Looks Like in the Retina |
|------|---------|----------------------------------|
| **N** | Normal | Healthy, even blood vessels, clear optic disc |
| **D** | Diabetes (Diabetic Retinopathy) | Tiny red dots (microaneurysms), cotton-wool spots |
| **G** | Glaucoma | Enlarged optic cup, thinning nerve fibre layer |
| **C** | Cataract | Hazy/blurry overall image due to cloudy lens |
| **A** | Age-related Macular Degeneration (AMD) | Yellow deposits (drusen) near the macula |
| **H** | Hypertension (Hypertensive Retinopathy) | Narrow/kinked arteries, flame-shaped haemorrhages |
| **M** | Myopia (Pathological) | Stretched, thin retina with lacquer cracks |
| **O** | Other Abnormalities | Any other condition not in the above list |

### Why Is This Important?

- **250+ million** people worldwide suffer from retinal diseases
- Many diseases (like diabetic retinopathy) are **symptom-free in early stages** — patients don't notice until vision loss is irreversible
- There is a severe **shortage of ophthalmologists**, especially in rural areas
- AI screening can **triage patients** quickly, flagging those who need urgent specialist attention

### What Is the ODIR-5K Dataset?

ODIR stands for **Ocular Disease Intelligent Recognition**. It is a structured ophthalmic database collected from multiple hospitals and medical centres in China.

- **~7,000 retinal fundus images** from ~3,500 patients
- Each patient has **left eye + right eye** images
- **Expert-annotated** with one or more of 8 disease labels
- Some patients have multiple simultaneous conditions

### Why Multi-Label Classification?

This is the **most important conceptual question** for your viva.

**The Problem with Multi-Class Classification:**

Multi-class classification assumes each input belongs to **exactly one** class. Like sorting mail into boxes — each letter goes into one and only one box.

But in medicine, **patients can have multiple diseases simultaneously**:

```
Patient A:  Diabetes ✓   Hypertension ✓   Cataract ✗   Glaucoma ✗
Patient B:  Diabetes ✗   Hypertension ✗   Cataract ✓   Glaucoma ✓
Patient C:  Normal ✓     (everything else ✗)
```

If we used multi-class classification, we would have to create a separate class for **every possible combination**:
- "Diabetes only"
- "Diabetes + Hypertension"
- "Diabetes + Hypertension + Cataract"
- ... and so on

With 8 diseases, that's 2⁸ = **256 possible combinations** — most with very few training examples. This is impractical.

**The Multi-Label Solution:**

Instead, we treat each disease as an **independent binary question**:
- "Is Diabetes present?" → Yes/No
- "Is Glaucoma present?" → Yes/No
- "Is Cataract present?" → Yes/No
- ... (8 separate Yes/No questions)

This is called **multi-label classification** and it is the correct approach for this medical problem.

> **Analogy 🎯**: Think of a doctor examining a patient. The doctor doesn't pick ONE disease from a dropdown list. Instead, the doctor checks a **checklist** — ticking every condition the patient has. That's exactly what multi-label classification does.

---

## 2. Machine Learning Pipeline

Here is the complete journey of an image through our system:

```
📷 Raw Retinal Image (from fundus camera)
        │
        ▼
┌───────────────────────┐
│  1. DATA LOADING      │  ← data_loader.py
│  Load CSV metadata    │     Reads the CSV file, matches image filenames
│  Map to image files   │     to file paths, creates multi-label vectors
│  Create label vectors │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  2. PREPROCESSING     │  ← preprocessing.py
│  Resize to 224×224    │     Standardises all images to same size
│  Normalise [0,1]      │     Converts pixel values to 0-1 range
│  (Optional CLAHE)     │     Enhances contrast for clearer features
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  3. DATASET PIPELINE  │  ← dataset_builder.py
│  tf.data pipeline     │     Efficient GPU-friendly data loading
│  Augmentation         │     Random flip/rotate/zoom for variety
│  Batching & Prefetch  │     Feeds data in groups of 32
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  4. MODEL             │  ← model.py
│  DenseNet121 backbone │     Pre-trained feature extractor
│  Custom head layers   │     Dense(512) → Dropout → Dense(8, sigmoid)
│  Sigmoid outputs      │     8 independent probabilities (0 to 1)
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  5. TRAINING          │  ← train.py
│  Phase 1: Frozen      │     Train only the new head layers (15 epochs)
│  Phase 2: Fine-tune   │     Unfreeze top 50 DenseNet layers (35 epochs)
│  Class weighting      │     Handle imbalanced disease frequencies
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  6. EVALUATION        │  ← evaluate.py
│  Precision, Recall    │     How accurate are the predictions?
│  F1-Score, ROC-AUC    │     Overall quality metrics
│  Confusion matrices   │     Per-disease error analysis
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  7. PREDICTION        │  ← predict.py
│  Load new image       │     Single-image inference
│  Get 8 probabilities  │     One probability per disease
│  Apply threshold      │     Convert probabilities to Yes/No
│  Return diagnosis     │     List of detected diseases
└───────────────────────┘
```

### What Happens at Each Stage — In Simple Terms

| Stage | What Happens | Why It's Needed |
|-------|-------------|-----------------|
| **Data Loading** | Read the CSV table that says which image has which diseases | We need structured data (not just loose images) |
| **Preprocessing** | Make all images the same size and brightness range | Neural networks need consistent, standardised inputs |
| **Dataset Pipeline** | Organise images into batches, add random distortions | Efficient training + prevents memorisation |
| **Model** | Extract visual patterns → map them to disease probabilities | This is the "brain" that learns to diagnose |
| **Training** | Show thousands of examples and adjust the model's "weights" | This is how the model learns |
| **Evaluation** | Measure how well the model performs on unseen data | We need to know if it's actually good |
| **Prediction** | Use the trained model on a brand-new patient image | This is the final product — the diagnosis |

---

## 3. File-by-File Breakdown

### 3.1 [src/data_loader.py](file:///c:/EDI_FINAL/src/data_loader.py) — The Data Foundation

**Purpose**: Load the ODIR-5K CSV file and prepare image paths with their corresponding multi-label vectors.

**What It Does Step by Step:**

1. **Reads the CSV file** (`full_df.csv`) using Pandas
2. **Validates** that required columns exist (`filename`, `N`, `D`, `G`, `C`, `A`, `H`, `M`, `O`)
3. **Builds full file paths** by joining the image directory with each filename
4. **Filters out missing images** — if an image listed in the CSV doesn't exist on disk, it's skipped with a warning
5. **Extracts label vectors** — the 8 disease columns become a NumPy array of shape [(N, 8)](file:///c:/EDI_FINAL/src/evaluate.py#58-145)

**Key Constants:**

```python
DISEASE_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
```
These are the column names in the CSV that correspond to each disease.

```python
DISEASE_NAMES = {
    'N': 'Normal', 'D': 'Diabetes', 'G': 'Glaucoma',
    'C': 'Cataract', 'A': 'Age-related Macular Degeneration',
    'H': 'Hypertension', 'M': 'Myopia', 'O': 'Other Abnormalities',
}
```
This maps short codes to human-readable names.

**What Is Multi-Hot Encoding?**

Each image's label is a vector of 8 binary values. For example:

```
Patient with Diabetes + Cataract:
   N  D  G  C  A  H  M  O
 [ 0, 1, 0, 1, 0, 0, 0, 0 ]
        ↑     ↑
    Diabetes  Cataract

Patient with only Normal findings:
 [ 1, 0, 0, 0, 0, 0, 0, 0 ]
   ↑
 Normal
```

This is called **multi-hot encoding** — unlike one-hot encoding (where only ONE position is 1), multiple positions can be 1 simultaneously.

> **Analogy 🎯**: Imagine a hotel check-in form with checkboxes:
> - ☐ Pool Access    ☐ Gym Access    ☐ Spa Access    ☐ Restaurant
>
> You can tick **any combination**. Multi-hot encoding is exactly this — a row of checkboxes where multiple can be ticked.

**Why This File Is Important**: Everything downstream — preprocessing, training, evaluation — depends on correct data loading. If labels are wrong, the model learns the wrong patterns.

---

### 3.2 [src/preprocessing.py](file:///c:/EDI_FINAL/src/preprocessing.py) — Image Standardisation

**Purpose**: Transform raw retinal images into a format suitable for neural networks.

**Three Key Operations:**

#### 1. Image Resizing (to 224×224 pixels)

```python
img = cv2.resize(img, (224, 224))
```

**Why?** DenseNet121 was designed for 224×224 inputs. All images must be the same size because neural networks have a fixed-size input layer. Raw retinal images can be 2000×2000+ pixels — too large and inconsistent.

> **Analogy 🎯**: Think of fitting photographs into a passport-size frame. Whether the original photo was taken with a professional camera (huge resolution) or a phone (small resolution), you crop and resize it to the exact standard size so it fits the form.

#### 2. Normalisation (pixel values to 0–1)

```python
img = img.astype(np.float32) / 255.0
```

Raw images have pixel values from 0 to 255. We divide by 255 to get values between 0 and 1.

**Why?** Neural networks work best with small, standardised numbers. Large values (0–255) cause unstable gradients during training, making learning slow or erratic. Values in [0, 1] keep the maths stable.

> **Analogy 🎯**: Converting temperatures from Fahrenheit (32–212) to a 0–1 scale. The information is the same, but the smaller scale makes calculations more manageable.

#### 3. CLAHE Contrast Enhancement (Optional)

```python
def apply_clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
```

**CLAHE** = Contrast Limited Adaptive Histogram Equalisation

This enhances the contrast of retinal images, making subtle features (like tiny blood vessel changes in diabetic retinopathy) more visible. It works on small regions (tiles) instead of the whole image, preventing over-enhancement.

**How It Works**: Converts the image to LAB colour space → enhances the L (lightness) channel → converts back. This improves contrast without distorting colours.

#### 4. TensorFlow-Native Preprocessing

```python
def tf_decode_and_resize(image_path, target_size=224):
```

This is the same resize + normalise operation, but done inside TensorFlow's computation graph. This is used during training for better performance because TensorFlow can parallelise these operations on the GPU.

---

### 3.3 [src/dataset_builder.py](file:///c:/EDI_FINAL/src/dataset_builder.py) — The Training Data Pipeline

**Purpose**: Build an efficient data pipeline that feeds preprocessed, augmented image batches to the model during training.

**The `tf.data` Pipeline:**

TensorFlow's `tf.data` API is like a **conveyor belt** in a factory:

```python
ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
```

This creates a dataset from arrays of image paths and labels.

```python
ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
```

**Shuffling**: Randomises the order of examples each epoch. Without shuffling, the model might see all "Normal" images first, then all "Diabetes" images, creating learning biases.

> **Analogy 🎯**: Imagine studying flashcards. If you always study them in the same order (all maths, then all science, then all history), you might memorise the order rather than the content. Shuffling forces genuine learning.

```python
ds = ds.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)
```

This loads and preprocesses images **in parallel** — multiple images are processed simultaneously for speed.

```python
ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

**Batching** (batch_size=32): Groups 32 images together. Instead of updating the model after every single image (noisy, slow), we average the learning signal over 32 images (stable, efficient).

**Prefetching**: While the GPU is training on the current batch, the CPU is already loading the next batch. This eliminates waiting time.

> **Analogy 🎯**: Prefetching is like a chef prepping ingredients for the next dish while the current one is in the oven. No idle time.

#### Data Augmentation

```python
_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),     # Mirror the image
    tf.keras.layers.RandomRotation(0.15),         # Rotate up to ±27°
    tf.keras.layers.RandomZoom(0.1),              # Zoom in/out by 10%
    tf.keras.layers.RandomContrast(0.2),          # Change brightness/contrast
])
```

**Why Augmentation?**

We only have ~7,000 images. Deep learning models have millions of parameters and need much more data. Augmentation artificially creates **variations** of existing images:

- **Random Horizontal Flip**: The retina looks similar when mirrored. This doubles the effective dataset.
- **Random Rotation (±27°)**: Fundus images can be taken at slightly different angles. The model should recognise diseases regardless of orientation.
- **Random Zoom (±10%)**: Simulates different camera distances.
- **Random Contrast (±20%)**: Simulates different lighting conditions when the photo was taken.

**Critical**: Augmentation is applied **only during training**, not during validation or testing. We need clean, unmodified images to honestly measure performance.

> **Analogy 🎯**: Imagine training a security guard to recognise faces. You show them photos of the same person in different lighting, from different angles, wearing glasses or not. This makes the guard more robust than only showing them one perfect headshot.

---

### 3.4 [src/model.py](file:///c:/EDI_FINAL/src/model.py) — The Deep Learning Architecture

**Purpose**: Build the neural network that converts images into disease predictions.

This is the **heart** of the project. Let's break it down piece by piece.

#### Architecture Overview

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────────────┐
│  DenseNet121 (Pre-trained)  │  ← Feature Extractor (learned from ImageNet)
│  121 layers deep            │     Extracts visual patterns from the image
│  Weights from 1.2M images   │     Detects edges → textures → shapes → objects
└─────────┬───────────────────┘
          │  (output: 7×7×1024 feature maps)
          ▼
┌─────────────────────────────┐
│  GlobalAveragePooling2D     │  ← Squeezes 7×7×1024 → 1024 numbers
└─────────┬───────────────────┘
          ▼
┌─────────────────────────────┐
│  BatchNormalization         │  ← Stabilises training
└─────────┬───────────────────┘
          ▼
┌─────────────────────────────┐
│  Dense(512, relu)           │  ← Learns disease-specific patterns
└─────────┬───────────────────┘
          ▼
┌─────────────────────────────┐
│  Dropout(0.5)               │  ← Randomly disables 50% of neurons (prevents overfitting)
└─────────┬───────────────────┘
          ▼
┌─────────────────────────────┐
│  Dense(8, sigmoid)          │  ← 8 independent probabilities (one per disease)
└─────────────────────────────┘
          │
          ▼
   [0.12, 0.87, 0.03, 0.92, 0.05, 0.11, 0.02, 0.08]
     N     D     G     C     A     H     M     O
```

#### What Is DenseNet121?

DenseNet (Densely Connected Network) is a CNN architecture where **every layer is connected to every other layer** in a feed-forward fashion. In a traditional network, Layer 3 only receives input from Layer 2. In DenseNet, Layer 3 receives input from **both Layer 1 and Layer 2**.

**Why DenseNet121 Specifically?**
- "121" refers to 121 layers deep — enough to learn complex patterns
- Dense connections encourage **feature reuse** — early features (like edges) are available to later layers
- Very parameter-efficient — fewer parameters than equivalent ResNet models
- Proven excellent performance on **medical imaging** benchmarks

#### What Is Transfer Learning?

Transfer learning means using a model that was **already trained on a different task** and adapting it to our task.

DenseNet121 was pre-trained on **ImageNet** — a dataset of 1.2 million everyday images (dogs, cars, buildings, etc.). Through this training, it learned to detect:
- Low-level features: edges, corners, textures
- Mid-level features: shapes, patterns, repeated structures
- High-level features: complex object parts

**The key insight**: Many of these features are **universal**. The ability to detect edges, blood vessel patterns, and texture changes is useful for retinal images too — even though the model has never seen a retina before.

We then add our own custom layers on top (the "head") that learn to map these general features to our specific 8 diseases.

> **Analogy 🎯**: Imagine hiring an experienced detective who has solved thousands of cases. You don't need to teach them from scratch how to look for clues, read body language, or analyse evidence. You just need to brief them on the **specific** case. Transfer learning is exactly this — we keep the detective's general skills (pre-trained weights) and only train them on our specific problem (retinal diseases).

#### Why Sigmoid Instead of Softmax?

This is a **critical** concept for your viva.

| | Softmax | Sigmoid |
|---|---------|---------|
| **Output** | Probabilities that sum to 1.0 | Independent probabilities (0 to 1 each) |
| **Assumption** | Input belongs to exactly ONE class | Input can belong to MULTIPLE classes |
| **Example** | [0.7, 0.2, 0.1] (must sum to 1) | [0.8, 0.9, 0.1, 0.7] (no sum constraint) |
| **Use case** | "Is this a cat, dog, or bird?" | "Does this patient have diabetes AND/OR glaucoma?" |

Softmax creates **competition** between classes — if the probability of "Diabetes" goes up, "Glaucoma" must go down. But a patient can have **both** diseases! Sigmoid treats each output independently, so "Diabetes" can be 0.9 and "Glaucoma" can also be 0.8.

#### Why Binary Cross-Entropy (BCE)?

Since each disease is an independent binary classification (present or not), we use **Binary Cross-Entropy** as the loss function — applied independently to each of the 8 outputs.

```
BCE = -[ y·log(p) + (1-y)·log(1-p) ]
```

Where:
- `y` = true label (0 or 1)
- `p` = predicted probability

If the model predicts 0.9 for a disease that is present (y=1), the loss is low (good). If it predicts 0.1 for a present disease, the loss is high (bad).

The total loss is the **average** of all 8 individual BCE losses.

#### Focal Loss (Alternative)

The code also implements **Focal Loss** — a modification of BCE that **down-weights easy examples** and focuses on hard, misclassified ones. This is especially useful for rare diseases where the model gets lazy and just predicts "not present" for everything.

```python
focal_weight = (1 - p_t)^gamma   # gamma=2 by default
```

When the model is very confident and correct (p_t close to 1), the focal weight becomes tiny → small loss. When the model is struggling (p_t close to 0), the weight is large → big loss → more learning.

#### What Is Dropout?

```python
layers.Dropout(0.5)
```

During training, **50% of neurons are randomly turned off** in each forward pass. This forces the network to not rely on any single neuron and distribute knowledge across all neurons.

> **Analogy 🎯**: In a football team, if one star player is always on the field, the team becomes dependent on them. If you randomly bench different players during practice, the whole team learns to perform well regardless of who's playing. That's dropout.

---

### 3.5 [src/train.py](file:///c:/EDI_FINAL/src/train.py) — The Training Process

**Purpose**: Train the model using a two-phase strategy with class weighting and smart callbacks.

#### Train-Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)
```

- **80%** of data → Training (the model learns from these)
- **20%** of data → Validation (the model is tested on these to monitor how well it generalises)

The validation set is like a **practice exam** — the model never trains on these images, so its performance on them reflects real-world performance.

`random_state=42` ensures the split is the same every time you run the code (reproducibility).

#### The Class Imbalance Problem

In the ODIR dataset, diseases are **not equally represented**:

```
Normal:       ~3,000 images    (very common)
Diabetes:     ~2,000 images    (common)
Myopia:       ~300 images      (rare)
Hypertension: ~200 images      (very rare)
```

Without correction, the model would learn to **always predict "Normal"** because it's right most of the time just by guessing the majority class. It would never learn to detect rare diseases.

**Solution — Class Weighting:**

```python
def compute_class_weights(labels):
    weight = num_negative / (num_positive + epsilon)
```

Rare diseases get **higher weights** — when the model misses a rare disease, it's penalised more heavily. Common diseases get lower weights.

> **Analogy 🎯**: In an exam, if 90% of questions are easy and 10% are hard, a student who only studies easy questions will score 90%. But if you make hard questions worth 10× more marks, the student is forced to study everything. That's class weighting.

#### Two-Phase Training Strategy

**Phase 1 — Frozen Backbone (15 epochs):**
- The DenseNet121 layers are **frozen** (their weights don't change)
- Only the custom head layers (Dense 512 + Dense 8) are trained
- Learning rate: 1e-4 (0.0001)
- This allows the head to learn basic mappings without destroying the pre-trained features

**Phase 2 — Fine-Tuning (35 epochs):**
- The top 50 layers of DenseNet121 are **unfrozen**
- The entire model is trained with a **much smaller** learning rate (1e-5)
- This gently adapts the pre-trained features to retinal-specific patterns

> **Analogy 🎯**: Imagine inheriting a well-organised kitchen (pre-trained DenseNet). In Phase 1, you only rearrange the top shelf (train the head) — you don't move the heavy appliances. In Phase 2, once you understand the layout, you make small, careful adjustments to the whole kitchen (fine-tune). You use gentle movements (low learning rate) to avoid breaking anything.

#### Key Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **Epochs** | 15 + 35 = 50 total | Number of complete passes through the training data |
| **Batch Size** | 32 | Number of images processed before one weight update |
| **Learning Rate** | 1e-4 (Phase 1), 1e-5 (Phase 2) | How big each weight adjustment step is |
| **Optimizer** | Adam | Adaptive learning rate optimiser (adjusts per-parameter) |
| **Validation Split** | 20% | Fraction of data reserved for validation |

**Epoch**: One epoch = the model has seen every training image exactly once. 50 epochs means the model sees the entire training dataset 50 times.

**Batch Size**: We don't feed all images at once (too much memory) or one at a time (too noisy). 32 is a balanced middle ground.

**Learning Rate**: Controls how much the model adjusts its weights after each batch. Too high → overshoots the optimal values. Too low → learns too slowly. Adam automatically adapts this during training, but we set the starting value.

#### Callbacks — Smart Training Control

The code uses three callbacks that automatically manage the training process:

**1. EarlyStopping:**
```python
EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True)
```
If the validation AUC doesn't improve for 5 consecutive epochs, **stop training** and restore the best weights. This prevents **overfitting** — where the model starts memorising training data instead of learning general patterns.

> **Analogy 🎯**: Like studying for an exam — at some point, more studying doesn't improve your score and might make you second-guess correct answers. EarlyStopping says "stop studying, you peaked 5 hours ago."

**2. ReduceLROnPlateau:**
```python
ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3)
```
If validation AUC stops improving for 3 epochs, **halve the learning rate**. This allows the model to make finer adjustments when it's close to the optimum.

> **Analogy 🎯**: When parking a car, you start with big steering movements. As you get closer to the perfect spot, you make tiny adjustments. ReduceLROnPlateau automatically shifts from "big movements" to "fine adjustments".

**3. ModelCheckpoint:**
```python
ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_auc', save_best_only=True)
```
Saves the model weights **only when** validation AUC improves. At the end of training, you have the best model ever achieved, not the last model (which might be worse due to overfitting).

---

### 3.6 [src/evaluate.py](file:///c:/EDI_FINAL/src/evaluate.py) — Measuring Model Performance

**Purpose**: Compute comprehensive metrics on a held-out test set and save reports.

#### Why Not Just Use "Accuracy"?

Accuracy = (correct predictions) / (total predictions)

Consider this scenario with imbalanced data:
- 95% of images are "not Hypertension"
- A model that **always predicts "not Hypertension"** achieves 95% accuracy!
- But it's completely useless — it never detects the disease

This is why we use more nuanced metrics:

#### Precision

```
Precision = True Positives / (True Positives + False Positives)
```

**"Of all the patients I flagged as having Disease X, how many actually have it?"**

High precision = few false alarms (important for avoiding unnecessary treatments)

> **Analogy 🎯**: A fire alarm with high precision rarely goes off by mistake. When it sounds, there's almost certainly a real fire.

#### Recall (Sensitivity)

```
Recall = True Positives / (True Positives + False Negatives)
```

**"Of all the patients who actually have Disease X, how many did I detect?"**

High recall = few missed cases (critical in medical diagnosis — missing a disease can be fatal)

> **Analogy 🎯**: A fire alarm with high recall catches almost every fire, but might occasionally go off for burnt toast too.

#### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The **harmonic mean** of precision and recall. It balances both concerns. An F1 of 0.83 means the model has a good balance of detecting diseases without too many false alarms.

#### ROC-AUC (Area Under the ROC Curve)

This measures how well the model **ranks** positive examples above negative examples across all possible thresholds. 

- **AUC = 1.0**: Perfect ranking (all positives ranked higher than all negatives)
- **AUC = 0.5**: Random guessing (no discrimination ability)
- **AUC = 0.78**: Good discrimination (our model's macro AUC)

> **Analogy 🎯**: Imagine sorting a deck of cards where red cards = disease. AUC measures how well the model sorts all red cards to the top of the deck. AUC = 1.0 means all red cards are on top. AUC = 0.5 means red cards are randomly mixed in.

#### Macro vs Micro Metrics

- **Macro**: Calculate the metric for each disease separately, then take the **unweighted average**. Treats all diseases equally regardless of how common they are.
- **Micro**: Pool all predictions together across all diseases, then calculate the metric. Gives more weight to common diseases.

**In our project**: Macro metrics are more informative because we care about **every** disease equally, even rare ones.

#### Prediction Threshold

The model outputs probabilities (e.g., 0.73 for Diabetes). We need to decide: is 0.73 "yes" or "no"?

The **threshold** makes this decision:
- Threshold = 0.5 → predict "yes" if probability ≥ 0.5
- Threshold = 0.3 → predict "yes" if probability ≥ 0.3 (** lower threshold, more sensitive**)

Our project uses **threshold = 0.3** because in medical screening:
- **Missing a disease (false negative) is worse than a false alarm (false positive)**
- A lower threshold makes the model more cautious — it flags more potential cases
- False positives can be caught by a human doctor in follow-up
- False negatives mean the patient goes undiagnosed

---

### 3.7 [src/predict.py](file:///c:/EDI_FINAL/src/predict.py) — Single-Image Inference

**Purpose**: Take one new retinal image and predict which diseases are present.

**The Inference Pipeline:**

```python
# 1. Load and preprocess the image
img = load_and_preprocess(image_path)       # Resize to 224×224, normalise to [0,1]

# 2. Add batch dimension
img_batch = np.expand_dims(img, axis=0)     # Shape: (224,224,3) → (1,224,224,3)

# 3. Get model predictions
preds = model.predict(img_batch)[0]         # Returns 8 probabilities

# 4. Apply threshold to get binary decisions
detected = [disease for prob in preds if prob >= threshold]
```

**Why `expand_dims`?** The model expects a batch of images (4D tensor: batch × height × width × channels). Even for a single image, we need to add a "batch dimension" of 1.

**Output Example:**
```
Probabilities:
  Normal                          0.1200
  Diabetes                        0.8700  ◀ (above threshold)
  Glaucoma                        0.0300
  Cataract                        0.9200  ◀ (above threshold)
  Age-related Macular Degeneration 0.0500
  Hypertension                    0.1100
  Myopia                          0.0200
  Other Abnormalities             0.0800

Detected Diseases:
  • Diabetes
  • Cataract
```

---

## 4. The Notebook: Experiments & Visualizations

The notebook [model_experiments.ipynb](file:///c:/EDI_FINAL/notebooks/model_experiments.ipynb) is the **experimentation lab** of the project. It demonstrates data analysis, model comparisons, and interpretability.

### 4.1 Dataset Exploration

#### Disease Distribution Analysis

The notebook visualises how many images belong to each disease category. The key finding is **severe class imbalance**:

| Disease | Approximate Count | Relative Frequency |
|---------|-------------------|-------------------|
| Normal (N) | ~3,000 | Very common |
| Diabetes (D) | ~2,000 | Common |
| Other (O) | ~1,500 | Common |
| Cataract (C) | ~400 | Moderate |
| Glaucoma (G) | ~400 | Moderate |
| Myopia (M) | ~300 | Rare |
| AMD (A) | ~250 | Rare |
| Hypertension (H) | ~200 | Very rare |

**Why some diseases are harder to detect**: Diseases with fewer training examples give the model less opportunity to learn their visual patterns. Hypertension (only ~200 examples) is much harder than Normal (~3,000 examples).

#### Label Cardinality

This shows how many diseases each patient has on average. Most patients have 1 disease, but some have 2 or 3 simultaneously — confirming why multi-label classification is necessary.

#### Co-occurrence Matrix

This shows which diseases tend to appear together. For example, Diabetes and Hypertension often co-occur because they share risk factors. Understanding co-occurrence helps the model learn correlations.

### 4.2 Image Visualization

The notebook displays sample retinal images for each disease category. This helps demonstrate:

- **Normal** eyes have clean, well-defined blood vessel patterns
- **Diabetic** eyes show microaneurysms (tiny red dots) and haemorrhages
- **Cataract** images appear hazy/blurry overall
- **Myopic** retinas appear stretched with a different optic disc shape

These visual differences are what the neural network learns to detect.

### 4.3 Baseline Model: DenseNet121

The notebook evaluates the primary DenseNet121 model (the one saved as `retinal_model.keras`). This serves as the **baseline** against which other approaches are compared.

**Why DenseNet121 as Baseline?**
- Proven track record in medical imaging research
- Dense connections enable excellent feature reuse
- Parameter-efficient (fewer weights than comparable architectures)
- Strong performance on the ImageNet benchmark
- Well-suited for **fine-grained visual distinctions** needed in retinal imaging

### 4.4 Model Comparison: DenseNet121 vs EfficientNetB3

The notebook trains and compares two different architectures:

| Metric | DenseNet121 | EfficientNetB3 |
|--------|-------------|----------------|
| **Macro F1** | **0.40** | 0.23 |
| **Micro F1** | **0.35** | 0.25 |
| **Precision** | **0.42** | 0.15 |
| **Recall** | 0.46 | **0.93** |
| **ROC-AUC** | **0.78** | 0.56 |
| **Hamming Loss** | **0.18** | 0.79 |

**Key Observations:**

- **DenseNet121 is clearly superior** in overall balanced performance (higher F1, higher AUC, lower hamming loss)
- **EfficientNetB3 has extremely high recall (0.93)** but abysmal precision (0.15) — it flags almost everything as positive, leading to massive false positives. This is not useful in practice.
- DenseNet121's dense connections are better suited for the **fine-grained texture differences** in retinal images
- EfficientNetB3, despite being designed for efficiency, didn't converge well with our limited dataset and training setup

**Why Architecture Differences Matter:**
- **DenseNet** reuses features from all previous layers — excellent for medical imaging where subtle textural details matter
- **EfficientNet** uses compound scaling (depth + width + resolution) — designed for efficiency but may need different training strategies on medical data

### 4.5 Ensemble Model

The notebook demonstrates **ensemble learning** — combining predictions from both DenseNet121 and EfficientNetB3:

```python
ensemble_probs = (densenet_probs + effnet_probs) / 2
```

| Metric | DenseNet121 | EfficientNetB3 | Ensemble (Avg) |
|--------|-------------|----------------|----------------|
| **Macro F1** | 0.40 | 0.23 | **0.43** |
| **Micro F1** | 0.35 | 0.25 | **0.41** |
| **Recall** | 0.46 | 0.93 | **0.60** |

**The ensemble improved F1 by ~8%** over DenseNet121 alone! By averaging predictions, the ensemble balances DenseNet121's conservative predictions with EfficientNetB3's aggressive ones.

> **Analogy 🎯**: Imagine two doctors reviewing the same X-ray. Doctor A is cautious (high precision, might miss subtle cases). Doctor B is aggressive (high recall, might over-diagnose). If they discuss and agree, their combined opinion is more reliable than either one alone. That's ensemble learning.

**Why Ensembles Work:**
- Different models make **different types of errors**
- Averaging smooths out individual model weaknesses
- The "wisdom of crowds" effect — multiple imperfect predictors combine to create a better predictor

### 4.6 Grad-CAM Visualization

**Grad-CAM** (Gradient-weighted Class Activation Mapping) is a technique that creates a **heatmap** showing which parts of the image the model focused on when making its prediction.

**How Grad-CAM Works (Simplified):**

1. Feed an image through the model
2. For a specific disease prediction, compute **gradients** flowing back to the last convolutional layer
3. These gradients indicate which spatial locations (pixels) were most important for that prediction
4. Create a heatmap where **warm colours (red) = important regions**, cool colours (blue) = ignored regions
5. Overlay the heatmap on the original image

**Why Is This Important?**

In medical AI, it's not enough for the model to say "this patient has Glaucoma." Doctors need to know **why** the model thinks so. If the Grad-CAM heatmap highlights the optic disc area (which is medically correct for Glaucoma), it increases trust in the model.

If the model focuses on irrelevant areas (like the image border), it suggests the model is learning **spurious patterns** rather than real medical features — a critical quality check.

> **Analogy 🎯**: Imagine a student who always gets the right answer on a maths test. You're suspicious, so you ask them to "show their work." If their working makes sense, you trust the answer. If they used a cheat sheet, you don't trust the answer even though it's correct. Grad-CAM is like asking the AI to "show its work."

**Why Explainability Matters in Medical AI:**
- Regulatory requirements (FDA/CE marking) demand explainable predictions
- Doctors won't trust a "black box" system
- Helps identify model failures before deployment
- Ethical obligation to ensure AI doesn't discriminate or learn biases

---

## 5. Understanding the Results

### Classification Report (Actual Project Results)

```
              precision    recall  f1-score   support

           N       0.46      0.26      0.34       392
           D       0.60      0.28      0.38       446
           G       0.26      0.52      0.35        84
           C       0.56      0.71      0.62        82
           A       0.14      0.58      0.22        59
           H       0.20      0.40      0.26        45
           M       0.88      0.78      0.83        67
           O       0.26      0.18      0.21       327
```

### Why Some Diseases Are Better Detected

**Best Performance — Myopia (F1 = 0.83):**
- Myopia has **very distinctive visual features** — the retina appears stretched and elongated
- Even with fewer samples (~300), the features are so clear that the model learns them well
- High precision (0.88) AND high recall (0.78) — a reliable detection

**Good Performance — Cataract (F1 = 0.62):**
- Cataracts cause a general haziness that's relatively easy to detect
- Good recall (0.71) — catches most cataract cases
- Reasonable precision (0.56)

**Poor Performance — Other Abnormalities (F1 = 0.21):**
- "Other" is a **catch-all** category — it includes many different, unrelated conditions
- The model cannot learn a single pattern for such diverse abnormalities
- Low precision (0.26) and low recall (0.18)

**Poor Performance — AMD (F1 = 0.22):**
- AMD features (drusen) are **very small** and subtle — hard to detect at 224×224 resolution
- Only 59 test samples — very limited data
- Low precision (0.14) — many false positives

### Overall Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| ROC-AUC (macro) | **0.784** | Good discrimination ability |
| F1 (macro) | 0.40 | Moderate average across all diseases |
| Hamming Loss | 0.18 | Only 18% of label predictions are wrong |
| Prediction Threshold | 0.3 | Lower than default 0.5 for higher sensitivity |

### Dataset Limitations

1. **Class imbalance** — huge disparity between common (Normal) and rare (Hypertension) diseases
2. **Small dataset** — ~7,000 images is small for deep learning; state-of-the-art research uses 100K+
3. **"Other" category** — a vague catch-all that hurts model performance
4. **Single source** — data from Chinese hospitals; may not generalise to other populations
5. **Image quality variability** — different cameras and operators create inconsistent image quality

---

## 6. Viva Questions & Answers

### Q1: Why did you use multi-label classification instead of multi-class?

> **Answer**: In real-world ophthalmology, patients often have **multiple diseases simultaneously**. For example, a diabetic patient may also have hypertensive retinopathy and early cataracts. Multi-class classification forces us to pick only ONE class per patient, which is medically incorrect. Multi-label classification treats each disease as an independent binary question — "Does this patient have Diabetes? Yes/No. Does this patient have Glaucoma? Yes/No." — allowing any combination of diseases to be detected simultaneously. This is achieved by using **sigmoid activation** (not softmax) and **binary cross-entropy loss**.

### Q2: Why did you choose DenseNet121 as the backbone?

> **Answer**: DenseNet121 was chosen for three reasons. First, its **dense connectivity** (where each layer receives input from ALL preceding layers) promotes feature reuse, which is excellent for medical imaging where subtle textural differences matter. Second, it's **parameter-efficient** — despite being 121 layers deep, it has fewer parameters than comparable ResNet architectures because of feature reuse. Third, DenseNet has demonstrated **state-of-the-art performance** on medical imaging benchmarks, including retinal image analysis in published research. Our experiments confirmed this — DenseNet121 significantly outperformed EfficientNetB3 on our dataset with a macro F1 of 0.40 vs 0.23.

### Q3: Why sigmoid instead of softmax?

> **Answer**: Softmax normalises outputs so they **sum to 1**, creating competition between classes — if one class probability goes up, others must go down. This assumes the input belongs to **exactly one** class. But in our medical problem, a patient can have Diabetes AND Cataract simultaneously. Sigmoid allows each output neuron to produce an **independent probability** between 0 and 1, with no constraint that they sum to 1. This means Diabetes can be 0.9, Cataract can be 0.85, and Normal can be 0.1 — all at the same time.

### Q4: What is transfer learning and why did you use it?

> **Answer**: Transfer learning means using a model pre-trained on a large dataset (ImageNet, 1.2 million images) and adapting it to our smaller medical dataset (~7,000 images). The pre-trained model has already learned to detect **universal visual features** — edges, textures, shapes, patterns — that are useful across many visual tasks. Instead of learning these from scratch with limited data (which would likely overfit), we leverage this existing knowledge and only train new layers that map these general features to our specific disease labels. This dramatically reduces training time and improves performance, especially when labelled medical data is scarce.

### Q5: What is the two-phase training strategy?

> **Answer**: In **Phase 1** (15 epochs), we freeze the entire DenseNet121 backbone and only train the custom classifier head (Dense layers). This prevents the randomly initialised head from sending chaotic gradients back through the pre-trained backbone and destroying its learned features. In **Phase 2** (35 epochs), we unfreeze the top 50 layers of DenseNet121 and fine-tune the entire model with a **10× lower learning rate** (1e-5 vs 1e-4). This allows gentle adaptation of the pre-trained features to retinal-specific patterns without catastrophic forgetting.

### Q6: Why is ROC-AUC a better metric than accuracy for this problem?

> **Answer**: Accuracy is misleading for imbalanced datasets. If 90% of samples are "not Hypertension," a model that always predicts "not Hypertension" achieves 90% accuracy but is useless. ROC-AUC measures how well the model **ranks** positive examples higher than negative examples across all possible thresholds. It's **threshold-independent** — it evaluates the quality of the probability scores, not just binary predictions. Our model achieved a ROC-AUC of 0.784, meaning it has good discrimination ability even if the raw F1 scores vary by disease.

### Q7: Why does class imbalance affect training?

> **Answer**: Neural networks optimise to minimise overall loss. If 90% of training examples are "Normal," the easiest way to minimise loss is to always predict "Normal" — the model never bothers learning rare disease patterns. We combat this with **class weighting** — rare diseases get higher loss weights, so misclassifying a Hypertension case is penalised much more heavily than misclassifying a Normal case. The model is forced to learn patterns for all diseases, not just the common ones.

### Q8: Why are some diseases harder to detect than others?

> **Answer**: Several factors explain this: (1) **Data quantity** — Myopia (300 samples) performs better than AMD (250) because Myopia has very distinctive features, while AMD has subtle ones. (2) **Visual distinctiveness** — Myopia causes dramatic retinal stretching that's easy to see; AMD has tiny drusen deposits that may only be a few pixels at 224×224 resolution. (3) **Category ambiguity** — "Other Abnormalities" is a catch-all for many unrelated conditions, making it nearly impossible to learn one coherent pattern. (4) **Inter-observer variability** — even human experts disagree on some diagnoses, introducing label noise.

### Q9: What is data augmentation and why is it important?

> **Answer**: Data augmentation artificially increases the diversity of training data by applying random transformations — horizontal flips, rotations (±27°), zoom (±10%), and contrast changes (±20%) — to existing images. This is crucial because: (1) Our dataset of ~7,000 images is small for deep learning, and augmentation effectively multiplies the dataset. (2) It reduces **overfitting** — the model sees slightly different versions of each image, preventing it from memorising exact pixel patterns. (3) It improves **generalisation** — the model learns to recognise diseases regardless of orientation, zoom level, or lighting conditions.

### Q10: What is Grad-CAM and why is explainability important?

> **Answer**: Grad-CAM (Gradient-weighted Class Activation Mapping) creates a heatmap showing which regions of the input image most influenced the model's prediction. It works by computing gradients from a specific output class back to the last convolutional layer, highlighting spatially important regions. Explainability is critical in medical AI because: (1) Doctors need to **trust** the system — seeing that the model focuses on clinically relevant regions (like the optic disc for Glaucoma) builds confidence. (2) It serves as a **quality check** — if the model focuses on image artifacts instead of medical features, we know it's learning the wrong patterns. (3) Regulatory bodies (FDA, CE) increasingly require explainability for AI medical devices.

### Q11: How does the ensemble model improve results?

> **Answer**: Our ensemble averages the probability outputs of DenseNet121 and EfficientNetB3. DenseNet121 is conservative (high precision, lower recall), while EfficientNetB3 is aggressive (high recall, very low precision). Averaging smooths out their individual weaknesses — the ensemble achieved a macro F1 of 0.43 versus 0.40 for DenseNet121 alone, an 8% improvement. This works because different architectures make **different errors**, and combining them reduces the overall error rate.

### Q12: What is the purpose of the prediction threshold?

> **Answer**: The model outputs probabilities (0 to 1) for each disease. The threshold converts these into binary decisions (disease present or not). We use **0.3 instead of the standard 0.5** because in medical screening, **missing a disease (false negative) is more dangerous than a false alarm (false positive)**. A lower threshold makes the model more sensitive — it catches more true cases at the cost of some false positives, which can be verified by a human doctor in follow-up examination.

### Q13: What improvements could be made to this project?

> **Answer**: Several improvements are possible:
> 1. **More data** — Using larger retinal datasets (EyePACS, APTOS) or synthetic data generation (GANs) to increase training samples
> 2. **Higher resolution** — Using 384×384 or 512×512 inputs to capture fine details like drusen and microaneurysms
> 3. **Better architectures** — Trying Vision Transformers (ViT) or ConvNeXt which have shown strong medical imaging results
> 4. **Per-class thresholds** — Instead of one global threshold (0.3), optimise a separate threshold for each disease based on its precision-recall curve
> 5. **Cross-validation** — Using 5-fold cross-validation instead of a single train/val split for more robust evaluation
> 6. **Attention mechanisms** — Adding attention layers to help the model focus on disease-relevant regions
> 7. **Multi-resolution input** — Processing images at multiple scales to capture both fine and coarse features

### Q14: What is binary cross-entropy and why is it used here?

> **Answer**: Binary cross-entropy measures the difference between predicted probabilities and true binary labels. For each of the 8 disease outputs, it computes: `BCE = -[y·log(p) + (1-y)·log(1-p)]`. When the model predicts correctly (high probability for present diseases, low for absent ones), BCE is small. When wrong, BCE is large, creating a strong learning signal. We use binary cross-entropy **instead of categorical cross-entropy** because each disease is an independent binary classification problem — the model makes 8 separate yes/no decisions, not one 8-way decision.

### Q15: How does the `tf.data` pipeline improve training speed?

> **Answer**: The `tf.data` pipeline provides three key optimizations: (1) **Parallel data loading** — `num_parallel_calls=AUTOTUNE` processes multiple images on different CPU cores simultaneously. (2) **Prefetching** — while the GPU trains on the current batch, the CPU loads and preprocesses the next batch, eliminating idle time. (3) **Vectorized batching** — images are grouped into tensors of 32 before GPU transfer, which is more efficient than transferring one image at a time. Together, these ensure the GPU is never waiting for data.

---

## 7. Real-World Analogies — Quick Reference

| Concept | Analogy |
|---------|---------|
| **Multi-label classification** | A doctor's checklist of conditions — tick all that apply |
| **Transfer learning** | Hiring an experienced detective — general skills transfer to new cases |
| **Sigmoid activation** | Individual light switches — each can be on or off independently |
| **Softmax activation** | A single dial — as you turn it towards one option, others decrease |
| **Dropout** | Randomly benching team players during practice — the whole team improves |
| **Data augmentation** | Showing flashcards at different angles, in different lighting |
| **Class weighting** | Making hard exam questions worth more marks |
| **EarlyStopping** | Knowing when to stop studying — more isn't always better |
| **ReduceLROnPlateau** | Fine parking adjustments after the big turn |
| **Prefetching** | Preparing the next dish while the current one is in the oven |
| **Grad-CAM** | Asking a student to "show their work" on a test |
| **Ensemble learning** | Two doctors consulting — their combined opinion is more reliable |
| **ROC-AUC** | How well you sort red cards to the top of a shuffled deck |
| **Batch normalisation** | Standardising ingredients before cooking — consistent results |
| **Fine-tuning** | Carefully rearranging an inherited, well-organised kitchen |
| **Prediction threshold** | Setting the sensitivity of a fire alarm |

---

## 8. 5–7 Minute Presentation Script

> **Instructions**: Practise this script. Memorise the flow, not the exact words. Speak confidently and use your own phrasing. Pause where indicated.

---

### Opening (30 seconds)

*"Good [morning/afternoon]. My project is titled **'Multi-Label Retinal Disease Detection using Deep Learning.'** I've built an AI system that analyses retinal fundus images — photographs of the back of the eye — and automatically detects which diseases are present. This addresses a critical healthcare challenge: over 250 million people worldwide suffer from retinal diseases, many of which are symptom-free in early stages and require expert ophthalmologists who are in severe shortage, especially in rural areas."*

### The Problem (45 seconds)

*"The unique challenge in retinal disease detection is that patients can have **multiple diseases simultaneously** — for example, diabetes and hypertension together. This makes it a **multi-label** classification problem, not a simple multi-class one. Multi-class classification forces picking one disease per patient, which is medically incorrect. Our approach uses **sigmoid activation** and binary cross-entropy loss to treat each of the 8 disease categories as an independent yes-or-no question."*

### Dataset & Preprocessing (45 seconds)

*"I used the **ODIR-5K dataset** — approximately 7,000 expert-annotated retinal images from hospitals in China, labelled for 8 conditions: Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, and Other Abnormalities. The dataset has **significant class imbalance** — Normal has about 3,000 images while Hypertension has only about 200. I addressed this through class weighting and data augmentation. Each image is preprocessed: resized to 224×224 pixels and normalised to a 0-to-1 range. I used TensorFlow's `tf.data` pipeline for efficient batching, shuffling, and prefetching."*

### Model Architecture (1 minute)

*"The core architecture is **DenseNet121 with transfer learning**. DenseNet121 was pre-trained on ImageNet — 1.2 million everyday images — and has learned to extract universal visual features like edges, textures, and shapes. I leverage this knowledge and add a custom classifier head: Global Average Pooling, Batch Normalisation, a Dense layer with 512 units, Dropout at 50% for regularisation, and a final Dense layer with 8 sigmoid-activated outputs — one probability per disease."*

*"Training follows a **two-phase strategy**: Phase 1 freezes the backbone and trains only the classifier head for 15 epochs. Phase 2 unfreezes the top 50 layers and fine-tunes the entire model with a 10× lower learning rate for 35 epochs. I used EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks to prevent overfitting and save the best model."*

### Experiments & Results (1.5 minutes)

*"In the experiments notebook, I compared **DenseNet121 against EfficientNetB3**. DenseNet121 achieved a macro F1 of 0.40 and ROC-AUC of 0.78, significantly outperforming EfficientNetB3 which had an F1 of only 0.23. I also tested an **ensemble** by averaging both models' predictions, which improved the macro F1 to 0.43 — an 8% improvement through model combination."*

*"Looking at per-disease performance, **Myopia** had the best F1 of 0.83 because it has very distinctive visual features — a stretched, elongated retina. **Cataract** achieved 0.62 due to its characteristic general haziness. Challenging diseases included AMD and Hypertension, which have very subtle features and limited training data."*

*"For model interpretability, I used **Grad-CAM** to visualise which image regions the model focuses on. This is crucial for clinical trust — we verified that the model attends to **medically relevant areas** like the optic disc for Glaucoma and the macula for AMD, rather than image artifacts."*

### Evaluation Approach (30 seconds)

*"I use precision, recall, F1-score, and ROC-AUC rather than simple accuracy, because accuracy is misleading for imbalanced medical datasets. I set the prediction threshold to 0.3 instead of 0.5 because in screening, **missing a disease is more dangerous than a false alarm**. The lower threshold improves recall at a slight cost to precision."*

### Conclusion (30 seconds)

*"In summary, this project demonstrates a practical deep learning pipeline for multi-label retinal disease detection. The system achieves an ROC-AUC of 0.78, with strong performance on distinctive conditions like Myopia and Cataract. Future improvements could include higher resolution inputs, Vision Transformers, per-class threshold optimisation, and training on larger datasets. Thank you."*

---

> [!TIP]
> **Final Tips for the Viva:**
> - If you don't know an answer, say: *"That's a great question. Based on my understanding..."* and reason through it logically
> - Always relate technical answers back to the **medical context** — examiners love seeing you understand WHY, not just WHAT
> - Admit limitations honestly — it shows maturity ("Our dataset is relatively small, and performance on rare diseases like AMD is a known limitation")
> - If asked about a concept you're unsure about, explain what you DO know and acknowledge what you need to explore further
