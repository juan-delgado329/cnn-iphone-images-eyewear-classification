# Convolutional Neural Networks for Eyewear Image Classification
### Using Real-World iPhone Images to Classify Fashion Eyewear SKUs

## Description

This project evaluates the feasibility of using Convolutional Neural Networks (CNNs) and transfer learning to classify eyewear products from iPhone images captured under variable lighting, backgrounds, and viewing conditions. The long-term goal is to develop a scalable mobile image recognition application to help field sales representatives identify and manage eyewear products more efficiently during customer visits.

A dataset of 420 real-world iPhone images representing 6 eyewear SKUs was collected and used to train and evaluate multiple CNN architectures, progressing from a scratch-built CNN to advanced transfer learning methods.

---

## Key Questions Answered

- Can a CNN model reliably classify specific eyewear SKUs from real-world iPhone images?
- How do scratch-built CNNs compare to transfer learning approaches for small datasets?
- Does a Hierarchical CNN (HCNN) architecture improve classification accuracy for visually similar fashion products?
- Which transfer learning model — MobileNetV2, VGG16, or VGG19 — performs best on this task?

---

## Tech Stack

- **Language:** Python
- **Environment:** Jupyter Notebook / Google Colab
- **Libraries:**
  - `TensorFlow / Keras` — model building, training, and evaluation
  - `MobileNetV2` — lightweight transfer learning backbone
  - `VGG16 / VGG19` — deep transfer learning for hierarchical classification
  - `matplotlib` — visualization of training results and validation examples
  - `NumPy` — data manipulation

---

## Data Acquisition

- **Total images:** 420 (70 images per SKU × 6 SKU classes)
- **Captured with:** iPhone camera
- **Conditions:** Variable backgrounds, lighting levels, and angles to simulate real sales environments
- **Angle:** 45-degree angle shots to expose both front and temple features of each eyewear piece
- **Locations:** 7 different home locations, 10 images per location per class

---

## Models Evaluated

### 1. Scratch CNN (Baseline)
- 3 convolutional layers with Max Pooling and ReLU activation
- Softmax output layer with Adam optimizer
- Result: Failed to generalize — essentially random guessing due to small dataset size

### 2. MobileNetV2 Transfer Learning
Three progressive experiments:
- **1st model:** Unfroze last 40 layers — partial learning, model stalling
- **2nd model:** Froze all layers + fine-tuning (adjusted learning rate, Global Average Pooling, reduced dense layers) — strong results
- **3rd model:** Unfroze last 20 layers for further fine-tuning — maintained performance with improved loss

### 3. Hierarchical CNN (HCNN) with VGG16 and VGG19
Inspired by Seo & Shin (2019), the HCNN classifies images at three hierarchical levels:
- **C1:** Brand
- **C2:** Product type (Sun or Optical)
- **Fine:** Specific SKU

---

## Results Summary

### MobileNetV2

| Metrics | Base CNN | 1st Model | 2nd Model | 3rd Model |
|---|---|---|---|---|
| Training Accuracy | 0.1941 | 0.2201 | 0.7238 | 0.7909 |
| Validation Accuracy | 0.1436 | 0.1905 | 0.8973 | 0.8817 |
| Validation Loss | 1.7941 | 1.9528 | 0.5371 | 0.4684 |

### HCNN — Training Metrics

| Training Metrics | Base HCNN | VGG16 HCNN | VGG19 HCNN |
|---|---|---|---|
| Overall Loss | 1.4350 | 0.0454 | 0.0933 |
| C1 Accuracy | 0.4524 | 0.9851 | 0.9315 |
| C2 Accuracy | 0.3333 | 0.9821 | 0.9524 |
| Fine Accuracy | 0.1756 | 0.9970 | 0.9702 |

### HCNN — Validation Metrics

| Validation Metrics | Base HCNN | VGG16 HCNN | VGG19 HCNN |
|---|---|---|---|
| Overall Loss | 1.4374 | 0.4132 | 0.4688 |
| C1 Accuracy | 0.4762 | 0.8571 | 0.8452 |
| C2 Accuracy | 0.3333 | 0.8571 | 0.8214 |
| Fine Accuracy | 0.1310 | 0.8452 | 0.8452 |

**Best performing model:** VGG16-HCNN with ~85% validation accuracy at the SKU level, with MobileNetV2 (2nd model) achieving ~90% validation accuracy — both demonstrating viable transfer learning approaches for real-world eyewear classification.

---

## Key Findings

- Scratch CNN models are insufficient for small, visually similar fashion datasets
- Transfer learning significantly improves performance even with limited data
- VGG16 outperformed VGG19 — the extra depth of VGG19 did not add discriminative value for eyewear shape classification
- The HCNN hierarchical structure effectively handles visually similar fashion classes
- Global Average Pooling outperformed Max Pooling for shape-based feature extraction
- Real-world iPhone images are a viable data source for CNN-based product classification

---

## References

1. Seo, Y., & Shin, K.-s. (2019). Hierarchical convolutional neural networks for fashion image classification. *Expert Systems with Applications, 116*, 328–339.
2. Iliukovich-Strakovskaia, A., et al. (2018). Non-personalized fashion outfit recommendations. *WorldCIST 2018*.
3. Eshwar S G, et al. (2016). Apparel classification using convolutional neural networks. *IEEE*.
4. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR 2018*.
5. Pardeshi, R., et al. (2025). Automated dried fish classification using MobileNetV2 and transfer learning. *IJACSA, 16(7)*.
6. Bhatnagar, S., et al. (2017). Classification of fashion article images using CNNs. *ICIIP 2017*.

---

## Author

Juan Delgado — Syracuse University, IST 691 (Deep Learning in Practice), 2025
