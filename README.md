# TamperVision-Lite

TamperVision-Lite is an image tampering detection and localization project built around the idea that digital manipulation introduces subtle forensic traces that are not always visible in standard RGB texture information. The project combines a forensic high-pass branch, a MobileNet-based visual branch, and a lightweight web interface to detect whether an image has been manipulated and to highlight suspicious regions.

This repository contains the full project pipeline:

- research and training experiments
- PyTorch model implementation
- TensorFlow/Keras inference pipeline for the final deployed model
- FastAPI backend for prediction
- browser-based frontend for interactive use
- evaluation scripts for image-level and localization metrics

---

## 1. Project Objective

The main goal is to classify an image as either authentic or tampered and, when possible, localize suspicious regions. In practice, tampering often affects local statistics, edge continuity, and compression artifacts. A model that only looks at raw RGB patterns may miss these traces. This project addresses that problem by fusing:

- RGB semantic features from MobileNet
- SRM forensic residual features that emphasize manipulation artifacts
- cross-attention fusion between the two feature streams
- Grad-CAM-based approximate localization for interpretability

This makes the project more useful than a simple binary image classifier because it does not only say “tampered,” it also tries to show where the suspicious area is.

---

## 2. Research Motivation

Image tampering can include copy-move, splicing, and region replacement. Such manipulations often preserve the overall image appearance, making them visually hard to detect. Standard CNNs are good at semantic content understanding, but they can fail to capture the forensic traces left behind by editing operations.

The project follows a research direction inspired by forensic analysis and hybrid CNN design:

1. RGB branch for semantic context and object appearance
2. SRM branch for noise residuals and high-frequency artifact detection
3. Fusion of both streams to combine appearance and forensic evidence
4. Localization using activation maps to highlight suspicious pixels/regions

This research approach is important because many tampering methods hide their edits in visually plausible content; thus, a model must reason beyond simple texture matching.

---

## 3. What Makes This Project Different

Compared with a baseline CNN-only model, this project introduces several important differences:

### 3.1 SRM-based forensic branch
The SRM (Steganalysis Rich Model) style filter is used to extract noise residual patterns. Manipulated regions often leave characteristic residual anomalies that are hard to detect with RGB-only networks. In the code, the SRM filter is implemented as a handcrafted high-pass filter that emphasizes edges and residual inconsistencies.

### 3.2 RGB + SRM fusion
The architecture fuses RGB features from MobileNet and SRM features from a separate forensic branch. The code does this at multiple scales:

- low
- mid
- high
- deep

This allows the model to combine fine-grained local artifacts with global semantic information.

### 3.3 Cross-attention fusion
The model uses a bidirectional cross-attention layer to allow one stream to guide the other. In other words, the RGB features and SRM features interact instead of being treated as separate fixed features.

### 3.4 Multi-task design
The model is designed not only for classification but also for localization. It predicts:

- image-level tampering probability
- localization mask / region map

This is more realistic than a purely classification-based detector.

### 3.5 Deployment-oriented web pipeline
The project is not limited to notebook experiments. It also includes:

- FastAPI prediction backend
- upload-based browser UI
- Grad-CAM visualization
- pseudo-mask generation
- bounding boxes over suspicious regions

This makes the project usable as an applied prototype rather than only as a research artifact.

---

## 4. Dataset

The research uses the CASIA2 dataset, which is a widely used benchmark for image tampering detection.

From the project analysis:

- Total valid images: 9,555
- Authentic images: 7,491
- Tampered images: 2,064
- Train / validation / test split: 70% / 15% / 15%

The dataset is split into image folders under the project structure and the training pipeline expects separate train/validation/test image and mask directories depending on the implementation stage.

---

## 5. Model Architecture

The project contains two major modeling views:

### 5.1 Research model (PyTorch)
The research version implements a hybrid architecture:

- RGB branch: MobileNetV3 feature extractor
- SRM branch: forensic residual extractor with SRM filter kernels
- Multi-scale fusion: combines RGB and SRM features
- Detection head: binary tampering classifier
- Localization decoder: reconstructs a mask at the input resolution

The model is implemented in:

- `models/mobilenetv3.py`
- `models/srm.py`
- `models/fusion.py`
- `models/cross_attention.py`
- `models/decoder.py`
- `models/tampering_model.py`

### 5.2 Final deployed model (TensorFlow/Keras)
The web backend loads a trained `.keras` model from `backend/model/` and uses a custom `SRMFilter` layer because the deployment model is based on a final trained checkpoint. This allows the system to do predictions in a browser-ready API flow.

The deployment logic in `backend/model.py` does the following:

1. reads uploaded image
2. converts it to RGB
3. resizes to 224 × 224
4. normalizes pixel values
5. runs model prediction
6. applies a business threshold
7. creates Grad-CAM heatmap
8. generates pseudo-mask and bounding boxes

---

## 6. Core Mathematical Idea

The model is based on the principle that real tampering leaves detectable traces beyond ordinary appearance features.

### 6.1 SRM filtering concept
SRM-like filters act as high-pass filters. The basic idea is:

- low-frequency natural image content is suppressed
- high-frequency edges and noise residuals are emphasized
- tampering boundaries often create irregular residual patterns

This helps expose manipulation even when the image looks visually natural.

### 6.2 Feature fusion
The joint representation is formed by combining RGB and forensic residual information at matching spatial scales:

- RGB feature map: $F_{rgb}$
- SRM feature map: $F_{srm}$
- fused feature: $F_{fused} = \phi(F_{rgb}, F_{srm})$

where $\phi$ is the fusion function combining projection and cross-attention.

### 6.3 Detection head
The detection head uses pooled fused features and a binary classifier:

- input: fused deep feature tensor
- pooling: global average pooling
- classifier: linear layers + ReLU + dropout + final sigmoid/logit layer

The final output is a tampering probability:

$$
P(tampered \mid x) = \sigma(W^T h(x) + b)
$$

where:

- $x$ = image input
- $h(x)$ = fused representation
- $W$ = classifier weights
- $b$ = bias
- $\sigma$ = sigmoid function

### 6.4 Localization using Grad-CAM
The backend uses Grad-CAM to create an activation heatmap. The Grad-CAM signal is computed from the gradient of the model output with respect to a convolutional feature map.

The standard Grad-CAM formula is:

$$
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}
$$

and the class activation map is:

$$
L^c_{GradCAM} = ReLU\left(\sum_k \alpha_k^c A^k\right)
$$

where:

- $A^k$ = feature map from the selected convolution layer
- $y^c$ = target class score
- $\alpha_k^c$ = importance weights for feature map $k$

The project then thresholds the generated heatmap to form a pseudo-mask and detects connected contours for bounding boxes.

---

## 7. Loss Functions and Training Objective

The research model is trained with a multi-task objective that combines detection and localization terms.

### 7.1 Binary cross entropy
For image-level tampering detection:

$$
\mathcal{L}_{det} = -\left[y \log(p) + (1-y)\log(1-p)\right]
$$

where $y \in \{0,1\}$ is the ground-truth label.

### 7.2 Dice loss
For localization or mask reconstruction, the project uses Dice loss:

$$
Dice = \frac{2 \cdot |X \cap Y| + \epsilon}{|X| + |Y| + \epsilon}
$$

and

$$
\mathcal{L}_{Dice} = 1 - Dice
$$

This helps when the region of tampering is often a small subset of the image and class imbalance is severe.

### 7.3 Total training objective
The code defines the total loss as a weighted combination:

$$
\mathcal{L}_{total} = w_{det}\mathcal{L}_{det} + w_{loc}\mathcal{L}_{loc}
$$

with the localization term using BCE + Dice:

$$
\mathcal{L}_{loc} = \mathcal{L}_{BCE} + \lambda \cdot \mathcal{L}_{Dice}
$$

This matches the project’s configuration in `traning/config.py` and `traning/loss.py`.

---

## 8. Evaluation Metrics

The project evaluates both image-level detection and localization quality.

### 8.1 Detection metrics
The code in `evaluation/detection_metrics.py` computes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- confusion matrix

The formulas are:

#### Accuracy
$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### Precision
$$
Precision = \frac{TP}{TP + FP}
$$

#### Recall
$$
Recall = \frac{TP}{TP + FN}
$$

#### F1-score
$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

### 8.2 Localization metrics
The project also has metrics for pixel-level segmentation quality in `evaluation/localization_metrics.py` using:

- IoU (Intersection over Union)
- Dice coefficient
- pixel precision
- pixel recall

#### IoU
$$
IoU = \frac{|X \cap Y|}{|X \cup Y|}
$$

#### Dice
$$
Dice = \frac{2|X \cap Y|}{|X| + |Y|}
$$

These are important because image-level classification alone does not tell whether the detected tampered region is correctly located.

---

## 9. Reported Accuracy and Results

The notebook shows a clear comparison between baseline and improved models.

### 9.1 Baseline MobileNetV2
From the notebook:

- Accuracy: 81.38%
- Precision: 62.87%
- Recall: 33.87%
- F1: 44.03%

### 9.2 SRM + MobileNetV2 Fusion
From the notebook:

- Accuracy: 80.82%
- Precision: 64.12%
- Recall: 40.32%
- F1: 47.62%

### 9.3 SRM + MobileNetV2 + Fine-Tuning
Final notebook summary:

- Accuracy: 80.89%
- Precision: 57.32%
- Recall: 45.48%
- F1: 50.72%

### 9.4 Discussion of metrics
The best tampering recall and F1 score are obtained in the fine-tuned hybrid model, which means the model becomes better at finding actual tampered cases rather than simply classifying many images as authentic. The project’s most important improvement is not necessarily raw accuracy alone, but the enhanced sensitivity to manipulated content.

The final comparison table in the notebook confirms this trend:

| Model | Accuracy | Precision | Tampered Recall | Tampered F1 |
|---|---:|---:|---:|---:|
| Baseline MobileNetV2 | 81.38% | 62.87% | 33.87% | 44.03% |
| SRM + MobileNetV2 Fusion | 80.82% | 64.12% | 40.32% | 47.62% |
| SRM + MobileNetV2 + Fine-Tuning | 80.89% | 57.32% | 45.48% | 50.72% |

Important note: the notebook reports different values depending on the exact training/evaluation stage, but the overall pattern is consistent: the hybrid forensic model improves tamper recall and F1 relative to the RGB-only baseline.

---

## 10. Business Decision Threshold in the Deployed App

The production backend in `backend/model.py` uses a stricter operational threshold than the notebook evaluation script.

It defines:

- default threshold = 0.20
- if probability < 0.20 => authentic
- else => tampered

This means the app intentionally uses a low probability cutoff to be conservative and reduce false negatives for authentic images, while still highlighting suspicious manipulations.

However, the evaluation metrics script uses a threshold of 0.5, which is the conventional binary threshold. That difference is important to understand when comparing notebook metrics with runtime app predictions.

---

## 11. Why the Project Is Useful in Practice

This project is useful because it goes beyond academic experimentation:

- it has a working backend API
- it accepts real uploaded images
- it generates heatmaps and pseudo-masks
- it produces suspicious-region bounding boxes
- it offers a browser interface for interactive testing

This makes it suitable for demo, research, and prototype deployment scenarios in digital forensics or image integrity checking.

---

## 12. Project Structure

```text
TamperVision-Lite-Full/
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── requirements.txt
│   └── model/
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── models/
│   ├── mobilenetv3.py
│   ├── srm.py
│   ├── fusion.py
│   ├── cross_attention.py
│   ├── decoder.py
│   ├── tampering_model.py
│   └── __init__.py
├── evaluation/
│   ├── detection_metrics.py
│   ├── localization_metrics.py
│   └── evaluate.py
├── prerocessing/
│   ├── image_preprocessing.py
│   ├── srm_filter.py
│   └── augmentation.py
├── traning/
│   ├── config.py
│   ├── loss.py
│   └── train.py
├── data/
├── dataset/
├── results/
├── checkpoints/
├── README.md
├── README_FRONTEND.md
├── REQUIREMENT.md
├── run_backend.bat
├── run_backend.sh
├── Copy_of_Capstone.ipynb
└── prepare_casia.py
```

---

## 13. How to Run the Project

### 13.1 Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 13.2 Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 13.3 Start the backend

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Or use:

```bash
run_backend.bat
```

### 13.4 Start the frontend

```bash
python -m http.server 5500 --directory frontend
```

Then open:

```text
http://127.0.0.1:5500
```

---

## 14. Important Research Notes

1. This is not a pure segmentation model trained on pixel-level ground truth for all tasks; the localization is derived from Grad-CAM and pseudo-mask generation.
2. The real application uses a business threshold of 0.20 in the backend, while the evaluation notebooks use standard classification thresholds.
3. The hybrid SRM + MobileNet strategy is more aligned with digital forensics than a plain RGB CNN.
4. The model is strongest at identifying tampered images with forensic residual anomalies and weaker at perfectly localizing every manipulated pixel.

---

## 15. Final Summary

TamperVision-Lite is a practical and research-oriented image tampering detection system that combines forensic signal processing and modern deep learning. Its key contribution is the fusion of SRM residual features with MobileNet-based visual features, which helps capture manipulation traces that ordinary RGB-only models often miss.

The project is different from a standard CNN baseline because it:

- includes SRM forensic processing
- uses multi-scale feature fusion
- adds cross-attention between RGB and SRM streams
- supports localization and explainability via Grad-CAM and bounding boxes
- provides a valid web application backend + frontend for end-to-end use

In short, this project is not just a simple classifier. It is an applied forensic deep-learning pipeline for tampering detection and approximate localization.

---

## 16. References and Project Sources Used

This README is based on the project’s actual source files and experiments, especially:

- `Copy_of_Capstone.ipynb`
- `backend/model.py`
- `backend/main.py`
- `models/tampering_model.py`
- `models/srm.py`
- `models/fusion.py`
- `models/cross_attention.py`
- `models/decoder.py`
- `evaluation/detection_metrics.py`
- `evaluation/localization_metrics.py`
- `traning/config.py`
- `traning/loss.py`

Together, these project files form the primary evidence base for the architecture, metric definitions, and reported results in this README.
