# III. Methodology

The RetinAI system is built as a hybrid clinical screening pipeline that takes a colour fundus image as input and produces a structured, interpretable diagnostic assessment as output. The design philosophy is straightforward: rather than treating retinal disease detection as a pure classification problem solvable by a single neural network, the system mirrors the multi-step reasoning process that ophthalmologists follow in practice. Three convolutional neural networks (EfficientNet-B3, DenseNet-121, and ConvNeXt-Tiny) form the classification backbone, and their per-class sigmoid outputs are fused through a per-class AUC-weighted soft-voting strategy. This ensemble runs alongside a classical computer vision branch that extracts clinically interpretable biomarkers from the raw image using deterministic algorithms. The outputs of both branches feed into a rule-based validation module that checks predictions against established clinical criteria, followed by a reasoning engine that incorporates patient demographics and reported symptoms to produce risk-stratified assessments.

The complete pipeline consists of eight sequential stages: image quality validation, CLAHE-based preprocessing, deep learning inference (three parallel models), classical feature extraction, ensemble fusion with per-class thresholding, rule-based clinical validation, clinical reasoning with risk stratification, and report generation with Grad-CAM visualisations. A design constraint that runs through every stage after inference is that model-derived confidence scores are never modified by downstream components. Validation flags, clinical interpretations, and patient context annotations are all additive; they enrich the diagnostic output without touching the predicted probabilities. This was a deliberate decision to preserve statistical calibration and ensure that the system's outputs remain fully auditable. The final output is a structured PDF report combining probability scores, spatial attention maps, clinical evidence, and contextualised recommendations.

## A. System Architecture

The pipeline is implemented in Python 3.10 using PyTorch 2.x, timm, OpenCV 4.x, Albumentations, and NumPy. Fig. 1 illustrates the overall system workflow. The first stage is an image quality gate that evaluates four criteria before allowing the image to proceed: blur score (Laplacian variance, threshold of 100.0), mean brightness (constrained to the range 40.0 to 220.0), contrast (standard deviation, threshold of 30.0), and optic disc detectability (requiring at least 0.1% of the image area). If any critical criterion is violated, specifically excessive blur, insufficient brightness, or absence of the optic disc, the image is rejected as ungradable with a structured error message. The rationale is simple: feeding a low-quality image into the ensemble will produce unreliable predictions, and it is better to reject the input and request a new capture than to generate a misleading diagnostic output.

After validation, the pipeline splits into two parallel branches. The first branch applies CLAHE enhancement and ImageNet normalisation, then passes the processed image through EfficientNet-B3, DenseNet-121, and ConvNeXt-Tiny independently. Each model produces an eight-dimensional sigmoid probability vector for the ODIR-5K categories: Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), Age-related Macular Degeneration (A), Hypertension (H), Myopia (M), and Other Abnormalities (O). These vectors are combined using per-class AUC-weighted soft voting and binarised using per-class optimised thresholds. The second branch operates on the raw RGB image, extracting a structured set of clinical biomarkers including lesion counts, vascular measurements, texture descriptors, and quality indicators using classical computer vision techniques.

The two branches converge in the rule-based validation module, which generates supporting, contradicting, or informational flags by checking whether the extracted clinical features are consistent with the ensemble predictions. These flags, together with the original probabilities and any patient-provided data (age, diabetes status, hypertension status, vision complaints), pass to the clinical reasoning engine. This module produces risk stratification (routine, semi-urgent, or urgent), evidence-backed explanations, and referral recommendations. An optional natural language generation component (Gemini 2.0 Flash, with Groq LLaMA 3.3 70B as fallback) can convert the structured output into patient-friendly language when API access is available; a template-based fallback handles offline scenarios. Grad-CAM heatmaps are generated for every condition exceeding its threshold, providing spatial evidence for the clinician. Throughout the entire post-inference pipeline, no component modifies the original ensemble probabilities.

## B. Dataset and Preprocessing

### 1) Dataset

The system is trained and evaluated on the ODIR-5K dataset, a multi-label ophthalmological benchmark containing 7,439 retinal fundus images from 3,500 patients captured using a variety of fundus cameras across multiple clinical sites. Each image carries eight binary labels for Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), Age-related Macular Degeneration (A), Hypertension (H), Myopia (M), and Other Abnormalities (O). Multiple pathologies can co-occur in a single image, which is what makes multi-label classification necessary rather than a simpler multi-class setup. The class distribution is heavily skewed: Normal and Diabetes together represent over 56% of positive labels, while Hypertension accounts for only about 2.7% (roughly 203 samples). This 10:1 imbalance between the most and least common classes has significant consequences for training and evaluation, and addressing it required both loss weighting and sampling strategies (described in Section III-B5). The dataset was chosen for its public availability (enabling reproducibility), its multi-disease annotation (matching the clinical screening use case), and its diversity of acquisition conditions (exposing the models to real-world imaging variability).

### 2) Data Splitting

Preserving the label distribution in both training and validation sets is essential when dealing with rare classes that have only a few hundred samples. The splitting strategy uses MultilabelStratifiedKFold from the iterative-stratification library with k=5 and random seed 42, selecting a single fold to produce an 80:20 split. This ensures that classes like Hypertension (2.7%) and Other Abnormalities (roughly 7.5%) appear in both sets in proportion to their overall prevalence rather than being concentrated in one partition by chance. If the iterative-stratification library is unavailable, a fallback uses scikit-learn's train_test_split stratified on each sample's rarest positive label, maintaining the same random seed. The resulting split yields approximately 5,951 training images and 1,488 validation images. All three ensemble models use the identical split, which is necessary for computing ensemble weights and thresholds on a consistent validation set.

### 3) Preprocessing

Every image goes through the same preprocessing pipeline during both training and inference (augmentation is applied only during training). First, CLAHE is applied to boost local contrast: the image is converted to LAB colour space, the luminance channel is processed with a clip limit of 2.0 and a tile grid size of 8x8, and the result is converted back to RGB. CLAHE was chosen over global histogram equalisation because it operates on local tiles, preserving fine retinal structures (vessels, small lesions, the optic disc) without amplifying noise in uniform regions. The image is then resized to 224x224x3 using bilinear interpolation, scaled to [0, 1], and normalised with ImageNet channel-wise statistics:

x_{\text{norm}} = \frac{x / 255.0 - \mu}{\sigma} \tag{1}

where μ=[0.485,0.456,0.406] and σ=[0.229,0.224,0.225]. The result is converted from a NumPy array of shape (224, 224, 3) to a channel-first tensor (3, 224, 224) using Albumentations' ToTensorV2, with a batch dimension added during inference to produce the final shape (1, 3, 224, 224).

### 4) Data Augmentation

The training augmentation pipeline uses Albumentations and applies a range of transformations designed to simulate the variability found in real-world fundus image acquisition. Horizontal flipping (probability 0.5) simulates left-right eye symmetry; vertical flipping (probability 0.2) handles occasional orientation differences. Brightness and contrast are perturbed within ±0.15 at probability 0.5, and hue (±10), saturation (±20), and value (±15) shifts at probability 0.3 simulate colour variability across cameras. Geometric transformations (±5% shift, ±10% scale, ±15° rotation, probability 0.4) account for positioning variation. Gaussian blur (kernel 3 to 5, probability 0.2) simulates mild focus issues, and coarse dropout (up to four holes of 20x20 pixels, probability 0.2) acts as a spatial regulariser, forcing the model to rely on distributed features rather than a single discriminative region. No augmentation is applied during validation or inference.

### 5) Class Imbalance Handling

Two strategies work together to address the class imbalance. The first is a per-class positive weighting scheme in the loss function: each class weight is the ratio of negative to positive samples, clipped to the range [1.0, 20.0]. This increases the loss contribution of rare classes like Hypertension and AMD so that their gradients are not drowned out by the abundant Normal and Diabetes samples. The second is a WeightedRandomSampler that assigns each training instance a sampling probability based on the inverse frequency of its rarest positive label. The reason for using both mechanisms is that loss weighting alone changes how much each sample contributes to the gradient but does not change how often rare samples appear in a training batch. Combining the two ensures that underrepresented conditions are both seen more frequently and weighted more heavily when they are seen.

## C. Model Architecture

### 1) Ensemble Design Rationale

The decision to use a three-model ensemble rather than a single network was motivated by a basic observation: different architectures capture different types of features, and for a dataset like ODIR-5K where the eight conditions have very different visual signatures, no single architecture is uniformly best. EfficientNet-B3 uses compound scaling, DenseNet-121 emphasises feature reuse through dense connectivity, and ConvNeXt-Tiny incorporates modern design elements borrowed from Vision Transformers. Combining them reduces the risk of correlated errors. Ensemble averaging also smooths out some of the overfitting that inevitably occurs when training on roughly 6,000 images, and per-class weighting lets the system lean more heavily on whichever model performs best for each specific condition. All three models share the same structural template: a backbone feature extractor, followed by dropout, followed by a fully connected layer that produces eight logits:

\text{Image} \xrightarrow{\text{Backbone}} \mathbb{R}^{d_{\text{feat}}} \xrightarrow{\text{Dropout}(p)} \xrightarrow{\text{Linear}(d_{\text{feat}}, 8)} \text{Logits} \in \mathbb{R}^{8} \tag{2}

### 2) EfficientNet-B3

EfficientNet-B3 uses MBConv blocks with squeeze-and-excitation attention, has approximately 12 million parameters, and produces features of dimension d_feat = 1536. It balances computational cost with representational capacity well enough to serve as the primary Grad-CAM model in the system; its intermediate feature maps produce more spatially coherent attention heatmaps than those from DenseNet-121 or ConvNeXt-Tiny. Dropout with p = 0.3 is applied before the classification layer.

### 3) DenseNet-121

DenseNet-121 (approximately 8 million parameters, d_feat = 1024) is the smallest model in the ensemble and the one least prone to overfitting on the limited ODIR-5K training set. Its dense connectivity pattern, where each layer concatenates the features of all preceding layers, encourages feature reuse and strong gradient flow. Dropout with p = 0.3 is applied in the classification head.

### 4) ConvNeXt-Tiny

ConvNeXt-Tiny (approximately 29 million parameters, d_feat = 768) is the largest and best-performing individual model, achieving a validation AUC of 0.8628. It uses 7x7 kernels, layer normalisation, GELU activations, and inverted bottleneck structures. Because of its higher capacity, dropout is set to p = 0.4 rather than 0.3 to compensate for the increased overfitting risk that comes with having roughly three times as many parameters as DenseNet-121.

### 5) Per-Class AUC-Weighted Soft-Voting Ensemble

Rather than averaging the three models equally, the ensemble weights each model's contribution to each class according to how well it discriminated that class on the validation set. For class c, model m receives a weight proportional to its validation AUC for that class, normalised so the three weights sum to 1:

w_{m,c} = \frac{\text{AUC}_{m,c}}{\sum_{j=1}^{3} \text{AUC}_{j,c}} \tag{3}

\hat{p}_c^{\text{ensemble}} = \sum_{m=1}^{3} w_{m,c} \cdot \hat{p}_{m,c} \tag{4}

where p̂_m,c is the sigmoid probability from model m for class c. The deployed weights are:

Class	EfficientNet-B3	DenseNet-121	ConvNeXt-Tiny
N (Normal)	0.3288	0.3248	0.3463
D (Diabetes)	0.3222	0.3274	0.3504
G (Glaucoma)	0.3265	0.3331	0.3404
C (Cataract)	0.3312	0.3334	0.3355
A (AMD)	0.3212	0.3297	0.3492
H (Hypertension)	0.3365	0.3286	0.3349
M (Myopia)	0.3353	0.3285	0.3362
O (Other)	0.3165	0.3201	0.3635

Table 1. Per-class AUC-weighted ensemble coefficients derived from validation performance.

ConvNeXt-Tiny consistently receives the highest weight, particularly for "Other Abnormalities" where its larger receptive field helps with the heterogeneous pathologies in that category. The weight differences are not dramatic for most classes, though, which means all three models contribute meaningfully and the ensemble is not simply deferring to ConvNeXt-Tiny.

### 6) Per-Class Optimal Thresholds

A single global threshold would force a compromise between sensitivity and specificity that cannot suit all eight conditions simultaneously. Instead, the ensemble probabilities are binarised using per-class thresholds found by grid search over [0.05, 0.95] in steps of 0.005, maximising the F1-score independently for each class:

Class	Optimal Threshold
N (Normal)	0.270
D (Diabetes)	0.330
G (Glaucoma)	0.835
C (Cataract)	0.780
A (AMD)	0.725
H (Hypertension)	0.550
M (Myopia)	0.945
O (Other)	0.545

Table 2. Per-class optimal decision thresholds obtained via validation F1-score maximisation.

The range from 0.270 (Normal) to 0.945 (Myopia) is wide. Low thresholds favour sensitivity; high thresholds favour specificity and are typical for conditions where the model is very confident when the disease is genuinely present but needs a strict cutoff to avoid false alarms.

## D. Training Procedure

### 1) Two-Phase Training Strategy

Each of the three models is trained independently on an NVIDIA Tesla T4 GPU (16 GB VRAM) with mixed-precision computation (float16), taking approximately 111.2 minutes total across all three. Training follows a two-phase protocol. In the first phase (frozen backbone warmup, 5 epochs), the pretrained backbone is frozen and only the randomly initialised classification head is trained. The point of this phase is to let the head learn reasonable outputs for the ODIR-5K label space before the backbone is exposed to gradients, because a randomly initialised head generates large, noisy gradients that can damage pretrained features if applied immediately. The AdamW optimiser is used with an initial learning rate of 1x10^-3 and cosine annealing to 1x10^-5. In the second phase, the full network is unfrozen and fine-tuned for up to 35 additional epochs at a lower learning rate, with cosine annealing continuing to 1x10^-7. Early stopping (patience of 8 epochs, monitoring validation AUC) restores the best checkpoint when training stalls.

### 2) Per-Model Hyperparameters

EfficientNet-B3 and DenseNet-121 both fine-tune at a learning rate of 3x10^-5 with weight decay of 1x10^-5. ConvNeXt-Tiny uses a higher learning rate of 5x10^-5 and weight decay of 1x10^-4 because its larger parameter count benefits from slightly more aggressive updates and stronger regularisation.

Hyperparameter	EfficientNet-B3	DenseNet-121	ConvNeXt-Tiny
timm model name	efficientnet_b3	densenet121	convnext_tiny
Feature dimension	1536	1024	768
Parameters (approx.)	12M	8M	29M
Dropout probability	0.3	0.3	0.4
Phase 1 (warmup) LR	1x10^-3	1x10^-3	1x10^-3
Phase 2 (fine-tuning) LR	3x10^-5	3x10^-5	5x10^-5
Weight decay	1x10^-5	1x10^-5	1x10^-4
Phase 1 epochs	5	5	5
Total epochs	40	40	40
Best validation AUC	0.8234	0.8255	0.8628

Table 3. Per-model hyperparameter configuration for ensemble training.

### 3) Shared Training Configuration

All models share the following settings to keep results comparable:

Hyperparameter	Value
Input resolution	224 x 224 x 3
Output activation	Sigmoid (per-class independent)
Number of classes	8
Optimizer	AdamW
Batch size	32
Total epochs (budget)	40 per model
Warmup epochs	5 per model
Early stopping patience	8 (monitor: val_auc, mode: max)
Phase 1 LR schedule	Cosine Annealing (10^-3 → 10^-5)
Phase 2 LR schedule	Cosine Annealing (LR → 10^-7)
Gradient clipping norm	1.0
Loss function	BCEWithLogitsLoss (pos_weight)
Mixed precision	float16 (AMP via GradScaler)
Random seed	42
Validation split	20% (MultilabelStratifiedKFold)
Ensemble method	Per-Class AUC-Weighted Soft-Voting

Table 4. Shared training configuration across all ensemble models.

### 4) Loss Function

Training uses BCEWithLogitsLoss, which combines sigmoid activation with cross-entropy in a single numerically stable operation. For N samples and C=8 classes:

\mathcal{L}_{\text{BCE}} = -\frac{1}{N \cdot C} \sum_{i=1}^{N} \sum_{c=1}^{C} w_c \left[ y_{ic} \log(\sigma(z_{ic})) + (1 - y_{ic}) \log(1 - \sigma(z_{ic})) \right] \tag{5}

where z_ic is the raw logit, σ is the sigmoid function, y_ic ∈ {0,1} is the ground truth, and w_c is the per-class positive weight:

w_c = \text{clip}\left(\frac{N_{\text{neg},c}}{N_{\text{pos},c} + \epsilon}, \; 1.0, \; 20.0\right) \tag{6}

with ε = 10^-6. The clip range [1.0, 20.0] prevents extremely rare classes from producing destabilising weight values while still providing a large amplification factor for their gradient contributions.

### 5) Training Stability Measures

Four mechanisms ensure stable training. Gradient clipping at a norm of 1.0 prevents explosion during fine-tuning (a real concern when unfreezing pretrained backbones). Mixed-precision computation (float16 via autocast and GradScaler) cuts memory use and speeds up training without loss of numerical precision in practice. Batches that produce NaN or infinite loss are automatically skipped. Deterministic seeding (random seed 42, fixed backend settings) makes runs reproducible.

### 6) Ensemble Threshold Optimisation

After individual training, all three models produce predictions on the shared validation set, which are combined using the per-class AUC-weighted voting described in Section III-C5. The optimal thresholds are then found by grid search, maximising F1-score per class. Performing this search on the ensemble output (rather than on individual model outputs) is important because the ensemble's probability distribution differs from any single model's, and thresholds optimised for a single model would be suboptimal for the fused predictions. The resulting thresholds are frozen and used unchanged in production.

## E. Clinical Feature Extraction and Rule-Based Clinical Validation

### 1) Clinical Feature Extraction Module

Running in parallel with the deep learning branch, this module computes 17 clinically grounded biomarkers from the raw RGB fundus image using OpenCV. The purpose is twofold: to provide interpretable evidence that complements the ensemble's probability scores, and to supply structured inputs for the rule-based validation module downstream. These features are meant to approximate what an ophthalmologist would observe during manual examination. They are not gold-standard measurements (the CDR estimation, in particular, uses brightness thresholding rather than segmentation, which is a known limitation), but their utility lies in cross-validation: when features and model predictions agree, clinical confidence increases; when they disagree, the contradiction is flagged for review.

#### a) Diabetic Retinopathy Features

Three biomarkers are extracted. Exudate count uses HSV filtering (hue 15-40, saturation 40-255, value 180-255) with morphological closing/opening and contour detection (minimum area 30 pixels squared). Haemorrhage count uses dual-range HSV filtering for red hues (0-10 and 160-180) with the same morphological pipeline. Microaneurysm count uses OpenCV's SimpleBlobDetector on the CLAHE-enhanced green channel, detecting small dark circular structures with area between 20 and 750 pixels squared, circularity greater than 0.5, convexity greater than 0.6, and inertia ratio above 0.4.

#### b) Glaucoma Features

The cup-to-disc ratio is estimated through intensity-based segmentation. The optic disc is found by thresholding a Gaussian-blurred grayscale image at the 90th percentile of brightness, refining with morphological operations (11x11 elliptical kernel, three closing and two opening iterations), and fitting a minimum enclosing circle. The optic cup uses a stricter threshold at the 95th percentile. CDR is the ratio of cup radius to disc radius, constrained to [0, 1].

#### c) Myopia Features

Four features: mean brightness of the green channel (overall illumination), texture variance of the grayscale image (structural variability), vessel visibility (mean absolute Laplacian response, indicating edge sharpness), and edge density (proportion of Canny-detected pixels, thresholds 50 and 150).

#### d) Cataract Features

Three features targeting the image degradation caused by lens opacity: blur score (Laplacian variance; lower means blurrier), contrast ratio (standard deviation of grayscale intensities), and haze index (mean divided by standard deviation of grayscale intensity; higher values indicate the uniform intensity distribution typical of cataract-affected images).

#### e) AMD Features

Features are extracted from the central macular region (central third of the image). Drusen count detects bright spots above the 92nd-percentile intensity threshold, filters for circular regions (area 10-500 pixels squared, circularity above 0.3) after morphological opening. Macular irregularity is the normalised standard deviation of the absolute Laplacian response within the macular region.

#### f) Hypertension Features

Two vascular features. The arteriole-to-venule (AV) ratio proxy is derived from a vessel mask (adaptive thresholding on the CLAHE-enhanced green channel), with widths estimated via distance transform and the ratio computed as the 25th percentile divided by the 75th percentile of non-zero widths. Vessel tortuosity is the ratio of arc length to chord length for contours exceeding 30 pixels, minus 1.0 to express deviation from straightness.

#### g) Normal and Other Features

For Normal: image clarity (Laplacian variance) and colour uniformity (mean of per-channel standard deviations). For Other Abnormalities: anomaly score (mean deviation of multi-orientation Gabor responses from their median) and irregularity density (fraction of pixels exceeding two standard deviations above the median response).

### 2) Rule-Based Clinical Validation

This module compares ensemble predictions with extracted clinical features to generate structured validation flags (supporting, contradicting, or informational). It does not modify predicted probabilities; it only annotates them. The rules are derived from established ophthalmological guidelines: ETDRS for diabetic retinopathy, AAO Preferred Practice Patterns and the ISNT rule for glaucoma, Keith-Wagener-Barker classification for hypertensive retinopathy, META-PM for myopia, AREDS and Beckman for AMD, and LOCS III for cataract.

The logic is straightforward. A diabetes prediction with zero detected haemorrhages, exudates, and microaneurysms triggers a high-severity contradicting flag (the model says DR, but no lesions were found). A glaucoma prediction with CDR exceeding 0.8 triggers a high-severity supporting flag (the model agrees with the clinical biomarker). Simultaneous predictions of diabetes and hypertension trigger an informational flag noting their known vascular co-morbidity. Concurrent predictions of Normal and any disease produce a high-severity contradiction since they are mutually exclusive in the ODIR framework.

## F. Evaluation Metrics and Explainability

### 1) Evaluation Metrics

Multiple metrics are used because no single number adequately characterises multi-label classification performance on an imbalanced dataset. Precision, recall, and F1-score are computed at both macro level (each class weighted equally) and micro level (aggregated across all classes):

\text{Precision} = \frac{TP}{TP + FP} \tag{7}

\text{Recall} = \frac{TP}{TP + FN} \tag{8}

F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \tag{9}

AUC-ROC serves as the primary threshold-independent metric and the basis for model selection and ensemble weighting:

\text{AUC} = \int_0^1 TPR(t)\, d(FPR(t)) \tag{10}

Hamming Loss captures the fraction of incorrect individual label predictions across all samples and classes:

\mathcal{H} = \frac{1}{N \cdot C} \sum_{i=1}^{N} \sum_{c=1}^{C} \mathbb{1}[\hat{y}_{ic} \neq y_{ic}] \tag{11}

Per-class 2x2 confusion matrices are also computed, which is particularly valuable for identifying weaknesses on rare conditions where aggregate metrics might look acceptable despite poor per-class performance. With the optimised ensemble and per-class thresholds, the system achieves a macro AUC-ROC of 0.8600, macro F1-score of 0.6280, micro F1-score of 0.5962, macro precision of 0.6192, macro recall of 0.6774, and Hamming Loss of 0.1439.

### 2) Grad-CAM Visual Explainability

Grad-CAM heatmaps are generated for each condition that exceeds its per-class threshold, using EfficientNet-B3 as the explainability backbone (chosen for the spatial coherence of its intermediate feature maps). For class c, the gradient of logit z_c with respect to feature maps A^k is computed, and channel-wise importance weights are obtained by global average pooling:

\alpha_k^c = \frac{1}{H \times W} \sum_i \sum_j \frac{\partial z_c}{\partial A_{ij}^k} \tag{12}

The heatmap is a ReLU-activated weighted sum of feature maps:

L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k \right) \tag{13}

The result is normalised to [0, 1], resized to 224x224, mapped to a JET colour scheme, and overlaid on the original image with alpha blending (0.6 original, 0.4 heatmap). Only conditions above threshold receive heatmaps, so the visual output is limited to clinically relevant predictions rather than showing attention maps for every possible category.

## G. Clinical Reasoning, Patient Context, and Deployment

### 1) Clinical Reasoning and Risk Stratification

This module converts ensemble probabilities into clinically actionable risk levels using the per-class thresholds with a safety margin:

Risk Level =
⎧ High Risk       if p ≥ 0.80
⎨ High Risk       if p ≥ θ_c + 0.15
⎩ Borderline      if p ≥ max(θ_c - 0.15, 0)
  Low Risk        otherwise
(14)

where p is the ensemble probability and θ_c is the class-specific threshold. The absolute floor at 0.80 exists for a specific reason: a disease like Myopia has a threshold of 0.945, which means a prediction of 0.85 would technically fall below threshold. But a probability of 0.85 represents strong model confidence, and downgrading it to "low risk" would be clinically irresponsible. The 0.80 floor catches these cases. An entropy-based out-of-distribution detection mechanism flags predictions as uncertain when the mean binary entropy across all classes exceeds 0.65, which indicates that the model is not confidently predicting anything and the input may lie outside the training distribution.

### 2) Patient Context Module and Symptom Stratification

This module collects patient age, diabetes status, hypertension status, and self-reported vision concerns (blurry vision, peripheral vision loss, floaters, and similar symptoms expressed in plain language). These are cross-referenced with model predictions to generate contextual annotations and assess symptom consistency on a three-tier scale: consistent match (symptoms align with detected conditions), partial match (some symptoms explained), and minimal or review-required (symptoms present but no corresponding detection). All contextual interpretations are strictly additive. They do not alter ensemble probabilities. Risk modifiers are grounded in clinical evidence: confirmed diabetes elevates DR risk (per ETDRS), age above 40 elevates glaucoma risk (per AAO guidelines), and absence of systemic conditions can reduce associated disease likelihood.

### 3) System Deployment and Clinical Reporting

RetinAI runs as a FastAPI web application with Jinja2 frontend interfaces for patients and clinicians. The ensemble models are loaded from a configuration file at startup and cached in memory; if ensemble loading fails, the system falls back to a single EfficientNet-B3 for operational continuity. PDF reports are generated using ReportLab and include ten sections: risk summary, findings table with probabilities, clinical interpretation, symptom assessment, and Grad-CAM visualisations. All artefacts are stored in a structured file system with SQLite metadata tracking. The pipeline supports both CPU and GPU execution with automatic mixed precision when a GPU is available. The complete workflow from image upload to report generation executes within a single request cycle.

In summary, the RetinAI system integrates per-class AUC-weighted ensemble classification, classical biomarker extraction, literature-backed rule-based validation, Grad-CAM explainability, and patient context interpretation into a single screening pipeline. Each post-inference component preserves the integrity of model predictions while adding progressively richer clinical context. The architecture is designed to approximate the multi-step, multi-source reasoning process that ophthalmologists use in practice rather than reducing diagnosis to a single forward pass through a neural network.
