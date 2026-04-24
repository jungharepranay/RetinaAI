# III. Methodology

The proposed RetinAI system is a hybrid clinical screening pipeline engineered to produce structured and clinically interpretable retinal disease assessments from colour fundus images. At its core, the system integrates a multi-model deep learning ensemble with classical computer vision-based biomarker extraction, rule-based clinical validation, gradient-based visual explainability, and a patient-context interpretation module. The classification backbone consists of three complementary convolutional architectures (EfficientNet-B3, DenseNet-121, and ConvNeXt-Tiny) whose per-class sigmoid outputs are combined through a per-class AUC-weighted soft-voting strategy. This ensemble formulation exploits the distinct inductive biases of compound-scaled networks, densely connected feature reuse, and modernised convolutional designs to improve generalisation across heterogeneous retinal pathologies. Unlike conventional end-to-end systems that rely solely on a single neural network, the proposed approach explicitly incorporates clinically grounded reasoning layers that emulate the multi-modal diagnostic workflow of ophthalmologists, a design choice motivated by the observation that classification accuracy alone is insufficient for clinical adoption without accompanying interpretive evidence.

The system accepts a single retinal fundus image and processes it through a sequential eight-stage pipeline. The first stage applies image quality validation to ensure that only gradable inputs propagate through downstream analysis. Following validation, the image undergoes CLAHE-based enhancement and standardised preprocessing before being passed simultaneously into two parallel branches: a deep learning inference branch and a classical feature extraction branch. In the inference branch, the preprocessed image is independently evaluated by the three backbone networks, producing multi-label sigmoid probability vectors that are fused via per-class AUC-weighted averaging. In parallel, the feature extraction branch computes a structured set of clinically relevant biomarkers (including lesion counts, vascular characteristics, and texture descriptors) directly from the raw image using deterministic computer vision techniques. The outputs of both branches are then integrated through a rule-based validation module that cross-references model predictions with extracted clinical evidence, followed by a reasoning engine that incorporates patient metadata such as age, systemic conditions, and reported symptoms to generate a risk-stratified clinical interpretation.

A key architectural constraint enforced throughout the pipeline is that model-derived confidence scores remain immutable after inference. All downstream modules operate strictly as read-only consumers of the ensemble outputs, producing additive annotations, validation flags, and interpretive summaries without modifying the underlying probabilities. This non-mutating design is adopted to ensure that statistical calibration is preserved while maintaining full auditability of the decision-making process; any modification to predicted probabilities by post-inference modules would compromise both calibration and reproducibility. The final stages of the pipeline include Grad-CAM-based visual explainability for spatial localisation of disease-relevant regions and optional natural-language explanation generation for patient-facing communication. Collectively, this multi-stage design transforms raw model predictions into a comprehensive clinical decision-support output that combines statistical inference, visual evidence, and contextual reasoning within a unified framework.

## A. System Architecture

The RetinAI pipeline is structured as a sequential eight-stage processing framework implemented in Python 3.10 using PyTorch 2.x, timm, OpenCV 4.x, Albumentations, and NumPy. Fig. 1 illustrates the overall system workflow, beginning with the ingestion of a retinal fundus image and progressing through validation, inference, and clinical reasoning stages. The pipeline first applies an image quality assessment module that evaluates four criteria: blur score using Laplacian variance with a threshold of 100.0, mean brightness constrained within the range 40.0–220.0, contrast measured via standard deviation with a threshold of 30.0, and optic disc detectability requiring at least 0.1% of the image area. If any critical condition (specifically excessive blur, insufficient brightness, or absence of the optic disc) is violated, the image is classified as ungradable and rejected with a structured diagnostic message. This early gating mechanism prevents unreliable inputs from propagating through the system and ensures downstream predictions remain clinically meaningful.

Following successful validation, the pipeline bifurcates into two parallel branches that operate concurrently. In the first branch, the image undergoes CLAHE-based enhancement and ImageNet normalisation before being passed into the ensemble of three convolutional neural networks: EfficientNet-B3, DenseNet-121, and ConvNeXt-Tiny. Each model independently produces an eight-dimensional sigmoid probability vector corresponding to the ODIR-5K disease categories: Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), Age-related Macular Degeneration (A), Hypertension (H), Myopia (M), and Other Abnormalities (O). These outputs are combined using a per-class AUC-weighted soft-voting mechanism, where each model's contribution to a given class is proportional to its validation AUC for that class. The fused probabilities are subsequently binarised using pre-optimised per-class thresholds and propagated unchanged to all downstream modules. In parallel, the second branch performs clinical feature extraction directly on the raw RGB image, generating a structured set of biomarkers including lesion counts, vascular measurements, texture descriptors, and quality indicators.

The outputs from both branches converge in a rule-based clinical validation module that cross-references ensemble predictions against extracted features to generate supporting, contradicting, or informational clinical flags. These flags, along with the original ensemble probabilities and patient-provided contextual data such as age, diabetes status, hypertension status, and vision complaints, are then processed by a clinical reasoning engine. This module produces a structured diagnostic interpretation with risk stratification levels (routine, semi-urgent, or urgent) alongside evidence-backed explanations and referral recommendations. An optional natural language generation component, implemented using Gemini 2.0 Flash or Groq LLaMA 3.3 70B as fallback, converts the structured output into patient-friendly explanations when API access is available, while a template-based fallback ensures operational continuity in offline scenarios. Grad-CAM heatmaps are generated for each condition exceeding its threshold, providing spatial visualisation of model attention over clinically relevant regions of the fundus image. Throughout this architecture, all post-inference components operate strictly as non-mutating layers, ensuring that original model confidence scores remain intact and preserving the integrity and auditability of the system.

## B. Dataset and Preprocessing

This section describes the dataset characteristics, data partitioning strategy, preprocessing pipeline, augmentation scheme, and class imbalance handling techniques employed in the RetinAI system. The design of this stage ensures that the input data distribution is standardised, clinically meaningful variations are preserved, and class imbalance effects are mitigated prior to model training. Each component is structured to maintain consistency across all three ensemble models while supporting generalisation under real-world imaging variability.

### 1) Dataset

The system is trained and evaluated on the Ocular Disease Intelligent Recognition (ODIR-5K) dataset, a multi-label ophthalmological benchmark comprising 7,439 retinal fundus images collected from 3,500 patients using diverse fundus imaging devices across multiple clinical environments. Each image is annotated with eight binary labels corresponding to Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), Age-related Macular Degeneration (A), Hypertension (H), Myopia (M), and Other Abnormalities (O). The dataset inherently reflects real-world clinical conditions in which multiple pathologies may co-occur within a single patient, thereby requiring a multi-label classification framework rather than independent binary detection tasks. The distribution of labels is highly imbalanced: Normal and Diabetes collectively account for over 56% of positive samples, while Hypertension represents only approximately 2.7% (around 203 samples), resulting in an approximate 10:1 imbalance ratio between the most and least frequent classes. This imbalance necessitates specialised training strategies such as class weighting and sampling adjustments, discussed in detail in Section III-B5. The dataset is selected for three reasons: its public availability enables reproducibility, its support for multi-disease annotation aligns with screening use cases, and its diversity in acquisition conditions enhances model adaptability across varying clinical scenarios.

### 2) Data Splitting

The dataset is partitioned into training and validation subsets using a multi-label stratified splitting strategy to preserve class distribution across all disease categories. When the iterative-stratification library is available, a MultilabelStratifiedKFold approach with k=5 and a fixed random seed of 42 is employed, from which a single fold is selected to produce an 80:20 split. This method is preferred over random splitting because it ensures that rare classes such as Hypertension (approximately 2.7%) and Other Abnormalities (approximately 7.5%) maintain proportional representation in both training and validation sets, a property that is critical for stable evaluation of minority-class performance. In scenarios where iterative stratification is unavailable, an approximate fallback strategy is implemented using scikit-learn's train_test_split, where stratification is performed based on the rarest positive label present in each sample while maintaining the same random seed for reproducibility. The resulting partition consists of approximately 5,951 training images and 1,488 validation images. This identical split is used across all three ensemble models to ensure statistical consistency during validation and to enable reliable computation of ensemble weights and thresholds.

### 3) Preprocessing

Each retinal image undergoes a standardised preprocessing pipeline applied consistently during both training and inference, with augmentation excluded during validation. Contrast-Limited Adaptive Histogram Equalisation (CLAHE) is applied first to enhance local contrast by converting the image from RGB to LAB colour space and operating on the luminance channel using a clip limit of 2.0 and a tile grid size of 8x8. CLAHE is preferred over standard histogram equalisation because it operates on local regions rather than the global intensity distribution, thereby improving the visibility of fine retinal structures such as blood vessels, exudates, and the optic disc while avoiding noise amplification in homogeneous areas. The image is then converted back to RGB and resized to a resolution of 224x224x3 pixels using bilinear interpolation, ensuring compatibility with all three backbone networks. Following resizing, pixel values are scaled to the range [0, 1] and standardised using ImageNet channel-wise statistics, aligning the input distribution with pretrained weights. The normalisation process is defined as:

x_{\text{norm}} = \frac{x / 255.0 - \mu}{\sigma} \tag{1}

where μ=[0.485,0.456,0.406] and σ=[0.229,0.224,0.225]. The processed image is converted from a NumPy array of shape (224, 224, 3) to a channel-first tensor of shape (3, 224, 224) using Albumentations' ToTensorV2, and a batch dimension is appended during inference to produce the final input tensor of shape (1, 3, 224, 224). This pipeline ensures consistent feature representation across all models and stable convergence during training.

### 4) Data Augmentation

To improve generalisation and simulate real-world variability in fundus image acquisition, a comprehensive augmentation pipeline is applied during training using the Albumentations library. The augmentation process includes multiple transformations applied probabilistically to each image. Random horizontal flipping with probability 0.5 simulates left-right eye symmetry, while vertical flipping with probability 0.2 accounts for occasional orientation variations. Illumination variability is introduced through brightness and contrast adjustments within ±0.15 at a probability of 0.5, and colour variability is simulated using hue (±10), saturation (±20), and value (±15) perturbations at a probability of 0.3. Geometric variability is incorporated through shift, scale, and rotation transformations with limits of ±5% shift, ±10% scaling, and ±15° rotation applied with probability 0.4. Gaussian blur with kernel sizes ranging from 3 to 5 and probability 0.2 simulates minor focus imperfections, while coarse dropout with a maximum of four holes of size 20x20 pixels and probability 0.2 acts as a spatial regulariser by forcing the model to rely on distributed features rather than localised cues. No augmentation is applied during validation or inference, where only deterministic preprocessing steps are used to ensure consistent evaluation.

### 5) Class Imbalance Handling

To address the severe class imbalance inherent in the dataset, two complementary strategies are employed during training. The first is a per-class positive weighting scheme applied within the loss function, where each class weight is computed as the ratio of negative to positive samples and clipped within the range [1.0, 20.0] to prevent instability from extremely rare classes. This weighting increases the contribution of underrepresented classes such as Hypertension and AMD during optimisation, ensuring that the gradient signal from rare positives is not overwhelmed by the dominant normal and diabetes classes. The second strategy uses a WeightedRandomSampler to rebalance mini-batches by assigning sampling probabilities to each training instance based on the inverse frequency of its rarest positive label. This dual approach is adopted because weighting alone adjusts the loss magnitude but does not change the frequency with which rare samples appear in training batches; combining both mechanisms ensures that rare disease samples are more frequently encountered during training without discarding majority-class data, thereby preserving the overall data distribution while improving sensitivity to minority classes. The output of this preprocessing and balancing stage feeds directly into the model training pipeline described in Section III-D.

## C. Model Architecture

This section describes the design of the multi-model ensemble classification backbone, including the architectural rationale, individual network configurations, ensemble fusion strategy, and threshold optimisation procedure. The architecture is specifically designed to balance representational diversity, class-wise adaptability, and reliable performance across heterogeneous retinal disease categories.

### 1) Ensemble Design Rationale

The classification component employs a three-model ensemble rather than a single deep network. This design is motivated by three considerations. First, architectural diversity reduces correlated prediction errors, as different network families capture complementary feature representations: EfficientNet exploits compound scaling, DenseNet emphasises feature reuse through dense connectivity, and ConvNeXt incorporates modern convolutional design principles inspired by Vision Transformers. A single network, regardless of capacity, is constrained by its particular inductive bias; combining three architecturally distinct networks mitigates this limitation. Second, ensemble averaging smooths the posterior probability distribution, thereby reducing overfitting to spurious correlations present in the relatively limited training dataset of approximately 6,000 images. Third, a per-class weighting mechanism enables the system to dynamically prioritise the model that performs best for a specific disease category, which is particularly important given the variability in classification difficulty across conditions. All three models follow a shared architectural template in which the input image is passed through a backbone feature extractor, followed by dropout regularisation and a fully connected layer producing an eight-dimensional logit vector, as formalised below:

\text{Image} \xrightarrow{\text{Backbone}} \mathbb{R}^{d_{\text{feat}}} \xrightarrow{\text{Dropout}(p)} \xrightarrow{\text{Linear}(d_{\text{feat}}, 8)} \text{Logits} \in \mathbb{R}^{8} \tag{2}

### 2) EfficientNet-B3

EfficientNet-B3 is based on the compound scaling methodology, which uniformly scales network depth, width, and input resolution using a single scaling coefficient. The architecture employs mobile inverted bottleneck convolution (MBConv) blocks combined with squeeze-and-excitation attention mechanisms to recalibrate channel-wise feature responses. With approximately 12 million parameters and a feature dimension of d_feat = 1536, it achieves a strong balance between computational efficiency and representational power. The classification head incorporates dropout with probability p = 0.3 to mitigate overfitting during fine-tuning. Of the three ensemble members, EfficientNet-B3 also serves as the primary Grad-CAM explainability model; this selection is motivated by its balanced performance and the stability of its intermediate feature representations, which produce more spatially coherent attention maps than those of the other two architectures.

### 3) DenseNet-121

DenseNet-121 employs a dense connectivity pattern in which each layer receives feature maps from all preceding layers via concatenation. This structure facilitates feature reuse, strengthens gradient flow, and improves learning efficiency for fine-grained visual patterns commonly present in retinal imagery. The model contains approximately 8 million parameters and produces a feature representation of dimension d_feat = 1024, making it the most lightweight component of the ensemble. Its parameter efficiency is particularly advantageous given the limited size of the ODIR-5K training corpus, as fewer parameters reduce the risk of overfitting on small datasets. The classification head uses dropout with probability p = 0.3 to ensure regularisation, consistent with EfficientNet-B3.

### 4) ConvNeXt-Tiny

ConvNeXt-Tiny represents a modernised convolutional architecture that incorporates several design elements inspired by Vision Transformers, including larger kernel sizes (7x7), layer normalisation, GELU activation functions, and inverted bottleneck structures. With approximately 29 million parameters and a feature dimension of d_feat = 768, it provides the highest individual performance among the three models, achieving a validation AUC of 0.8628. Due to its larger capacity, a higher dropout probability of p = 0.4 is used in the classification head to improve generalisation and reduce overfitting. The stronger regularisation is necessary because the ratio of trainable parameters to training samples is substantially higher for ConvNeXt-Tiny than for the other two models in the ensemble.

### 5) Per-Class AUC-Weighted Soft-Voting Ensemble

The outputs of the three models are combined using a per-class AUC-weighted soft-voting mechanism that assigns higher importance to models exhibiting superior discriminative performance for a given disease. This approach is preferred over uniform averaging because the three architectures do not perform equally across all disease categories; for instance, ConvNeXt-Tiny may excel on heterogeneous conditions captured by the "Other Abnormalities" class, while DenseNet-121 may offer stronger discrimination for conditions requiring fine-grained texture analysis. For each class c, the weight assigned to model m is proportional to its validation AUC for that class and is normalised such that the weights sum to unity, as defined in equation (3). The final ensemble probability for each class is then computed as a weighted sum of individual model probabilities, as shown in equation (4):

w_{m,c} = \frac{\text{AUC}_{m,c}}{\sum_{j=1}^{3} \text{AUC}_{j,c}} \tag{3}

\hat{p}_c^{\text{ensemble}} = \sum_{m=1}^{3} w_{m,c} \cdot \hat{p}_{m,c} \tag{4}

where p̂_m,c denotes the sigmoid probability predicted by model m for class c. This formulation enables adaptive weighting across disease categories, ensuring that the ensemble exploits the strengths of each model dynamically. The per-class weights used during deployment, derived from validation AUC scores, are presented below.

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

The weight distribution indicates that ConvNeXt-Tiny consistently contributes the largest share across all classes, reflecting its superior overall AUC performance. The difference is particularly pronounced for the "Other Abnormalities" category, where its larger receptive field and modernised architecture provide improved capability in modelling heterogeneous pathologies. It is worth observing, however, that the weight differentials are relatively narrow for most classes, suggesting that no single model dominates the ensemble decisively and that all three contribute meaningfully to the final prediction.

### 6) Per-Class Optimal Thresholds

The ensemble probabilities are converted into binary predictions using per-class optimal thresholds determined through exhaustive grid search on the validation set. The search is conducted over the interval [0.05, 0.95] with a step size of 0.005, optimising the F1-score independently for each class. This per-class approach is adopted rather than a single global threshold because class prevalence, model calibration, and decision boundary characteristics differ substantially across the eight disease categories; a global threshold would inevitably sacrifice sensitivity on rare classes or specificity on common ones. The resulting thresholds are shown below.

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

The wide variation in threshold values, ranging from 0.270 for Normal to 0.945 for Myopia, reflects differences in model calibration and the trade-off between sensitivity and specificity across disease categories. Lower thresholds prioritise sensitivity, ensuring detection of normal cases, while higher thresholds reduce false positives in conditions where the model exhibits high confidence when the disease is truly present. The thresholds produced by this procedure feed directly into both the binarisation step in the inference pipeline and the risk stratification module described in Section III-G.

## D. Training Procedure

This section details the training methodology employed for all three ensemble models, including the transfer learning strategy, optimisation configuration, loss formulation, and stability mechanisms. The training process is designed to ensure efficient convergence, tolerance to class imbalance, and reproducibility across runs while maintaining consistency across all backbone architectures.

### 1) Two-Phase Training Strategy

All three models are trained independently using a two-phase transfer learning protocol with mixed-precision computation (float16) on an NVIDIA Tesla T4 GPU with 16 GB VRAM, resulting in a total training time of approximately 111.2 minutes. The rationale for a two-phase approach, rather than training the full network from the outset, is to prevent the randomly initialised classification head from generating large, noisy gradients that would destabilise the pretrained backbone features during early training. In the first phase, referred to as frozen backbone warmup, the backbone parameters are kept non-trainable while only the classification head is optimised. This phase spans five epochs and enables the randomly initialised linear head to adapt to the ODIR-5K label space without perturbing pretrained ImageNet features through large gradient updates. The AdamW optimiser is used with an initial learning rate of 1x10^-3, and a cosine annealing schedule reduces the learning rate to 1x10^-5 over the warmup duration. In the second phase, the entire network (including both backbone and classification head) is unfrozen and jointly fine-tuned for up to 35 additional epochs. During this stage, a reduced learning rate is applied to enable gradual adaptation of both low-level and high-level feature representations to retinal imagery, while a cosine annealing schedule further decreases the learning rate to 1x10^-7. Early stopping with a patience of 8 epochs monitors validation AUC and restores the best-performing model weights when no improvement is observed.

### 2) Per-Model Hyperparameters

The three backbone models employ slightly different hyperparameter configurations to account for differences in architecture size and capacity. EfficientNet-B3 and DenseNet-121 both use a fine-tuning learning rate of 3x10^-5, while ConvNeXt-Tiny uses a higher value of 5x10^-5 due to its larger parameter space and modern design. Weight decay is set to 1x10^-5 for EfficientNet-B3 and DenseNet-121, and 1x10^-4 for ConvNeXt-Tiny to provide stronger regularisation commensurate with its greater model capacity. All models are trained for a maximum of 40 epochs, including 5 warmup epochs, and employ dropout probabilities consistent with their architectural definitions. The complete per-model configuration is presented below.

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

In addition to model-specific parameters, a common training configuration is applied across all three models to ensure consistency and comparability of results. The input resolution is fixed at 224 x 224 x 3 pixels, and the output layer employs independent sigmoid activations for each of the eight classes to support multi-label classification. The AdamW optimiser is used with a batch size of 32, and training is conducted for a maximum of 40 epochs per model with early stopping based on validation AUC. Cosine annealing schedules are applied during both warmup and fine-tuning phases, while gradient clipping with a norm of 1.0 is enforced to stabilise updates. Mixed-precision training is enabled using automatic mixed precision (AMP) with GradScaler to improve computational efficiency without compromising numerical stability. A fixed random seed of 42 ensures deterministic behaviour, and a 20% validation split is maintained using multi-label stratification. The ensemble combination strategy is consistent across all models, employing per-class AUC-weighted soft voting as described previously.

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

The training objective is defined using binary cross-entropy with logits, which integrates the sigmoid activation and cross-entropy computation into a numerically stable formulation. For a batch of N samples and C=8 classes, the loss is defined as:

\mathcal{L}_{\text{BCE}} = -\frac{1}{N \cdot C} \sum_{i=1}^{N} \sum_{c=1}^{C} w_c \left[ y_{ic} \log(\sigma(z_{ic})) + (1 - y_{ic}) \log(1 - \sigma(z_{ic})) \right] \tag{5}

where z_ic represents the raw logit for class c of sample i, σ(·) is the sigmoid function, y_ic ∈ {0,1} denotes the ground-truth label, and w_c is the class-specific positive weight. These weights are computed as:

w_c = \text{clip}\left(\frac{N_{\text{neg},c}}{N_{\text{pos},c} + \epsilon}, \; 1.0, \; 20.0\right) \tag{6}

where N_pos,c and N_neg,c denote the number of positive and negative samples for class c, respectively, and ε = 10^-6 prevents division by zero. Clipping the weights within the range [1.0, 20.0] ensures that extremely rare classes do not introduce instability during optimisation while still receiving substantially amplified gradient contributions relative to common classes.

### 5) Training Stability Measures

Several mechanisms are incorporated to ensure stable and efficient training across all models. Gradient clipping with a maximum norm of 1.0 is applied to prevent gradient explosion during fine-tuning, a concern that is particularly acute when unfreezing deep pretrained backbones in the second training phase. Mixed-precision training using float16 arithmetic via PyTorch's autocast and GradScaler reduces memory consumption and accelerates computation without compromising numerical stability. Batches producing non-finite loss values (such as NaN or infinite values) are automatically skipped to prevent corrupted parameter updates. Deterministic seeding is enforced through fixed random states and backend configurations, ensuring full reproducibility of results across multiple runs. These measures collectively ensure that the training procedure is both computationally efficient and numerically sound.

### 6) Ensemble Threshold Optimisation

After training, each model generates sigmoid probability predictions on the shared validation set, which are then combined using the per-class AUC-weighted ensemble strategy described in Section III-C5. The resulting ensemble probabilities are used to determine optimal decision thresholds for each class through exhaustive grid search over the interval [0.05, 0.95] with a step size of 0.005. The objective is to maximise the F1-score independently for each class, ensuring that threshold selection aligns with the final ensemble behaviour rather than individual model outputs. This post-training optimisation step is critical for balancing sensitivity and specificity across imbalanced disease categories and directly influences the final classification performance of the system. The optimised thresholds produced at this stage are frozen and deployed unchanged in production inference.

## E. Clinical Feature Extraction and Rule-Based Clinical Validation

### 1) Clinical Feature Extraction Module

This module operates in parallel with the deep learning inference branch and is responsible for extracting interpretable, clinically relevant biomarkers directly from the raw RGB fundus image using classical computer vision techniques implemented in OpenCV. The primary objective of this component is twofold: to provide transparent and verifiable clinical evidence that complements model predictions, and to supply structured inputs for the downstream rule-based validation module. A total of 17 disease-specific features are computed across all eight ODIR categories, encompassing lesion detection, vascular morphology, texture characteristics, and global image quality indicators. These features are designed to approximate clinically observable signs used by ophthalmologists during manual diagnosis, thereby bridging the gap between black-box model outputs and interpretable medical reasoning. It is important to acknowledge that these classical vision-based approximations do not replace gold-standard clinical measurements; rather, they serve as automated proxies whose concordance (or discordance) with model predictions provides an additional layer of interpretive evidence.

#### a) Diabetic Retinopathy Features

Three key biomarkers are extracted for diabetic retinopathy detection. Exudate count is computed using HSV colour space filtering with hue values between 15 and 40, saturation between 40 and 255, and value between 180 and 255, followed by morphological closing and opening operations and contour detection with a minimum area threshold of 30 pixels squared. Hemorrhage count is detected using a dual-range HSV filter targeting red hues in the ranges 0–10 and 160–180, combined with the same morphological processing pipeline to isolate candidate regions. Microaneurysm count is identified using OpenCV's SimpleBlobDetector, configured to detect small, dark, circular structures with area between 20 and 750 pixels squared, circularity greater than 0.5, convexity greater than 0.6, and inertia ratio above 0.4, applied on the CLAHE-enhanced green channel to maximise lesion contrast.

#### b) Glaucoma Features

The primary feature for glaucoma assessment is the cup-to-disc ratio (CDR), which is estimated using intensity-based segmentation. The optic disc is first identified by thresholding a Gaussian-blurred grayscale image at the 90th percentile of brightness, followed by morphological refinement using an 11x11 elliptical kernel with three closing and two opening iterations, and fitting a minimum enclosing circle. The optic cup is subsequently segmented using a higher threshold at the 95th percentile. The ratio of the cup radius to the disc radius is then computed and constrained within the range [0, 1], providing a clinically meaningful indicator of glaucomatous damage.

#### c) Myopia Features

Four features are extracted to characterise myopic changes. These include the mean brightness of the green channel, which reflects overall retinal illumination; texture variance computed from the grayscale image, capturing structural variability; vessel visibility measured as the mean absolute response of the Laplacian operator, indicating edge prominence; and edge density calculated as the proportion of pixels detected by the Canny edge detector with thresholds 50 and 150 relative to the total image area. Together, these features provide a quantitative representation of retinal texture and vascular clarity associated with myopic degeneration.

#### d) Cataract Features

Cataract-related features focus on image degradation caused by lens opacity. Blur score is computed as the variance of the Laplacian of the grayscale image, where lower values indicate increased blur. Contrast ratio is measured as the standard deviation of grayscale pixel intensities, reflecting loss of contrast in hazy images. Haze index is defined as the ratio of the mean to the standard deviation of grayscale intensity, where higher values correspond to more uniform intensity distributions typical of cataract-affected images.

#### e) AMD Features

Age-related macular degeneration is characterised using features extracted from the central macular region, defined as the central third of the image. Drusen count is detected as bright spots using a 92nd-percentile intensity threshold, followed by morphological opening and filtering for circular regions with area between 10 and 500 pixels squared and circularity greater than 0.3. Macular irregularity is computed as the normalised standard deviation of the absolute Laplacian response within the macular region, capturing structural disruption associated with degeneration.

#### f) Hypertension Features

Two vascular features are computed to capture hypertensive changes. The arteriole-to-venule (AV) ratio proxy is derived from the vessel mask obtained via adaptive thresholding on the CLAHE-enhanced green channel, where vessel widths are estimated using a distance transform and the ratio of the 25th percentile to the 75th percentile of non-zero widths is calculated. Vessel tortuosity is quantified by computing the ratio of arc length to chord length for significant vessel contours exceeding 30 pixels in length and subtracting 1.0 to express deviation from straightness.

#### g) Normal and Other Features

For normal classification, global image clarity is quantified using Laplacian variance, and colour uniformity is measured as the mean of per-channel standard deviations. For other abnormalities, two features are computed: an anomaly score defined as the mean deviation of multi-orientation Gabor filter responses from their median, and irregularity density calculated as the fraction of pixels exceeding two standard deviations above the median response. These features capture heterogeneous patterns that do not fall into predefined disease categories.

### 2) Rule-Based Clinical Validation

The rule-based validation module integrates ensemble predictions with extracted clinical features to generate structured validation flags categorised as supporting, contradicting, or informational. This module is strictly non-invasive with respect to model outputs; it does not modify predicted probabilities but instead produces auxiliary annotations that enhance interpretability and clinical trust. The validation rules are derived from established ophthalmological guidelines, including ETDRS for diabetic retinopathy, AAO Preferred Practice Patterns for glaucoma, the ISNT rule, the Keith-Wagener-Barker classification for hypertensive retinopathy, the META-PM classification for pathological myopia, the AREDS and Beckman classifications for AMD, and the LOCS III system for cataract grading.

For each predicted disease, the module evaluates threshold-based conditions on the corresponding clinical biomarkers to determine consistency with established clinical criteria. A diabetes prediction accompanied by zero detected hemorrhages, exudates, and microaneurysms, for instance, results in a high-severity contradicting flag, indicating inconsistency with expected pathological signs. Conversely, a glaucoma prediction with a cup-to-disc ratio exceeding 0.8 generates a high-severity supporting flag, reinforcing the model's prediction. In addition to single-disease validation, cross-disease interactions are also considered: simultaneous predictions of diabetic retinopathy and hypertension trigger an informational flag highlighting shared vascular manifestations and known clinical co-morbidity, while concurrent predictions of Normal and any disease category produce a high-severity contradiction due to their mutual exclusivity within the ODIR grading framework. These rule-based validations provide an additional layer of clinical reasoning that complements statistical model outputs, enabling more reliable and interpretable diagnostic assessments. The validation flags produced by this module are consumed by the clinical reasoning engine described in Section III-G.

## F. Evaluation Metrics and Explainability

### 1) Evaluation Metrics

This section outlines the quantitative metrics used to evaluate model performance on the held-out validation set, ensuring a comprehensive assessment of multi-label classification behaviour. Given the imbalanced and multi-label nature of the ODIR-5K dataset, multiple complementary metrics are employed to capture different aspects of predictive performance. These metrics are computed at both the macro level, where each class contributes equally regardless of prevalence, and the micro level, where predictions are aggregated across all classes. This dual evaluation strategy ensures that performance on rare conditions is not overshadowed by dominant classes while also reflecting overall system effectiveness.

Precision is defined as the proportion of predicted positive instances that are true positives, as given by equation (7). Recall, also referred to as sensitivity, measures the proportion of actual positive instances that are correctly identified, as shown in equation (8). The F1-score combines precision and recall into a single harmonic mean, balancing false positives and false negatives, as expressed in equation (9):

\text{Precision} = \frac{TP}{TP + FP} \tag{7}

\text{Recall} = \frac{TP}{TP + FN} \tag{8}

F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \tag{9}

To provide a threshold-independent evaluation, the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) is used as the primary metric for training, model selection, and ensemble weighting. AUC measures the model's ability to distinguish between positive and negative classes across all possible decision thresholds and is defined as:

\text{AUC} = \int_0^1 TPR(t)\, d(FPR(t)) \tag{10}

where the true positive rate TPR(t) and false positive rate FPR(t) vary as functions of the classification threshold t. This metric is particularly suitable for imbalanced datasets, as it remains insensitive to class distribution. Hamming Loss is employed to quantify the fraction of incorrectly predicted labels across all samples and classes, capturing both false positives and false negatives in a multi-label setting:

\mathcal{H} = \frac{1}{N \cdot C} \sum_{i=1}^{N} \sum_{c=1}^{C} \mathbb{1}[\hat{y}_{ic} \neq y_{ic}] \tag{11}

where 1[·] is the indicator function. Unlike strict accuracy, Hamming Loss evaluates errors at the individual label level, making it more appropriate for multi-label classification tasks. Class-wise 2x2 confusion matrices are computed independently for each disease category, enabling detailed analysis of true positives, false positives, true negatives, and false negatives, which is particularly valuable for identifying weaknesses in rare classes. Using the optimised ensemble and per-class thresholds, the system achieves a macro AUC-ROC of 0.8600, macro F1-score of 0.6280, micro F1-score of 0.5962, macro precision of 0.6192, macro recall of 0.6774, and a Hamming Loss of 0.1439.

### 2) Grad-CAM Visual Explainability

To enhance interpretability and clinical trust, Gradient-weighted Class Activation Mapping (Grad-CAM) is employed to generate spatial heatmaps that highlight regions of the fundus image contributing most strongly to each predicted disease class. The implementation targets the final convolutional layer of the EfficientNet-B3 backbone, which serves as the primary explainability model. EfficientNet-B3 is selected for this role because of its balanced performance and stable feature representations, which produce more spatially coherent attention maps than those obtained from the deeper ConvNeXt-Tiny or the more compact DenseNet-121. For a given class c, the gradient of the corresponding logit z_c with respect to the feature maps A^k is computed via backpropagation, and channel-wise importance weights are obtained using global average pooling, as defined in equation (12):

\alpha_k^c = \frac{1}{H \times W} \sum_i \sum_j \frac{\partial z_c}{\partial A_{ij}^k} \tag{12}

These weights are then used to compute the Grad-CAM heatmap as a weighted combination of feature maps followed by a rectified linear activation, as shown in equation (13):

L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k \right) \tag{13}

The resulting heatmap is normalised to the range [0, 1], resized to match the input resolution of 224x224 pixels, and mapped to a JET colour scheme. It is subsequently overlaid on the original fundus image using alpha blending with weights of 0.6 for the original image and 0.4 for the heatmap, producing an interpretable visual representation of model attention. Heatmaps are generated only for disease classes whose predicted probabilities exceed their respective optimal thresholds, ensuring that visual explanations correspond to clinically relevant predictions. This mechanism allows clinicians to verify whether the model focuses on anatomically meaningful regions such as the optic disc in glaucoma, the macular region in AMD, or vascular structures in diabetic and hypertensive retinopathy, thereby improving transparency and facilitating clinical validation of AI-generated outputs.

## G. Clinical Reasoning, Patient Context, and Deployment

### 1) Clinical Reasoning and Risk Stratification

This module synthesises ensemble predictions, rule-based validation outputs, and patient-specific contextual data to generate a structured and clinically interpretable assessment. The objective is to transform probabilistic outputs into actionable diagnostic insights through risk stratification and evidence-based reasoning. Risk levels are determined relative to per-class optimal thresholds with an additional safety margin, ensuring consistent interpretation across disease categories. The risk formulation is defined as:

Risk Level =
⎧ High Risk       if p ≥ 0.80
⎨ High Risk       if p ≥ θ_c + 0.15
⎩ Borderline      if p ≥ max(θ_c - 0.15, 0)
  Low Risk        otherwise
(14)

where p denotes the ensemble probability and θ_c represents the class-specific optimal threshold. The inclusion of an absolute safety floor at p ≥ 0.80 ensures that highly confident predictions are always classified as high risk, even for diseases with elevated thresholds such as glaucoma and myopia. This design choice is motivated by the clinical imperative that a prediction exceeding 80% confidence should never be downgraded to borderline status regardless of the threshold configuration. An entropy-based out-of-distribution (OOD) detection mechanism is incorporated, flagging predictions as uncertain when the mean binary entropy across all classes exceeds 0.65, indicating a lack of confident model consensus. The inclusion of entropy-based OOD detection is adopted because standard sigmoid outputs do not distinguish between confident negative predictions and genuinely uncertain ones; high entropy across all classes signals that the input image may lie outside the training distribution. This reasoning framework ensures that diagnostic outputs are not only statistically grounded but also clinically interpretable and safe for downstream decision-making.

### 2) Patient Context Module and Symptom Stratification

The patient context module augments model predictions with structured demographic and symptomatic information to provide personalised and clinically meaningful recommendations. Inputs include patient age, diabetes status, hypertension status, and self-reported vision concerns expressed in simplified, patient-friendly language such as blurry vision, peripheral vision loss, or visual disturbances like floaters. These inputs are cross-referenced with model predictions and clinical findings to generate contextual annotations and risk modifiers. Symptom consistency is evaluated using a three-tier stratification scheme in which findings are categorised as consistent match when symptoms fully align with detected conditions, partial match when only some symptoms are explained, and minimal or review-required when symptoms are present without corresponding model-detected pathology. These categories are presented through intuitive visual indicators and integrated into the final clinical report.

All contextual interpretations remain strictly additive and do not influence the underlying ensemble predictions or confidence values, a constraint that is enforced to prevent subjective patient-reported data from altering calibrated model outputs. Risk modifiers are derived from established clinical evidence: a confirmed diabetes history increases the risk weighting for diabetic retinopathy based on ETDRS findings, while age greater than 40 elevates glaucoma risk according to AAO guidelines and epidemiological studies. Conversely, absence of systemic conditions may reduce associated disease likelihood. This module enables the generation of tailored recommendations such as routine diabetic screening intervals or escalation to advanced imaging modalities when risk factors are present, thereby enhancing the clinical relevance of the system's output.

### 3) System Deployment and Clinical Reporting

The RetinAI system is deployed as a web-based clinical decision support platform using the FastAPI framework, providing a RESTful backend integrated with Jinja2-based frontend interfaces for both patients and clinicians. The backend architecture is organised into distinct route groups handling patient interactions, clinician workflows, and authentication processes. The trained ensemble models are dynamically loaded from a configuration file specifying architecture types, checkpoint paths, dropout parameters, ensemble weights, and threshold values, and are cached in memory to ensure efficient inference. In scenarios where ensemble loading fails, the system defaults to a single EfficientNet-B3 model to maintain operational continuity.

The system generates a comprehensive clinical screening report in PDF format using the ReportLab library, structured into ten sections including a risk assessment summary, key findings table with probability scores, detailed clinical interpretation, and symptom assessment mapping. Grad-CAM visualisations are embedded within the report to provide spatial evidence supporting each prediction, and all generated artefacts are stored and served through a structured file system integrated with a SQLite database for metadata tracking. The pipeline supports both CPU and GPU execution, with automatic mixed-precision enabled when GPU resources are available. The entire workflow, from image upload to report generation, is executed within a single request cycle, enabling real-time deployment in clinical and telemedicine environments.

In summary, the RetinAI methodology presents a comprehensive hybrid screening framework that integrates a per-class AUC-weighted ensemble of EfficientNet-B3, DenseNet-121, and ConvNeXt-Tiny with classical computer vision-based biomarker extraction, literature-driven rule-based validation, and context-aware clinical reasoning. The system incorporates threshold-optimised multi-label classification, entropy-based uncertainty detection, Grad-CAM visual explainability, and patient-specific contextual interpretation to produce structured, evidence-backed diagnostic outputs. Each stage of the pipeline is explicitly designed to preserve the statistical integrity of model predictions while progressively enriching them with clinically interpretable information and actionable insights. This layered architecture enables the system to closely emulate the multi-modal diagnostic reasoning process of expert ophthalmologists, thereby enhancing both reliability and practical applicability in real-world retinal disease screening scenarios.
