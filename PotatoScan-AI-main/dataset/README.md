# Dataset Information

## Overview

This project uses a combined potato leaf disease dataset created from two publicly available sources:

### PlantVillage
PlantVillage contains high-quality images captured under controlled laboratory conditions with uniform backgrounds and lighting.

### PlantDoc
PlantDoc contains real-world field images with varying lighting conditions, backgrounds, image quality, and camera angles.

Combining both datasets helps improve model generalization across controlled and real-world environments.

---

## Classes

The final dataset contains three classes:

- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

---

## Dataset Composition

| Class | Images |
|---------|---------:|
| Potato – Early Blight | ~3,425 |
| Potato – Late Blight | ~3,425 |
| Potato – Healthy | ~2,430 |
| **Total** | **9,280** |

---

## Dataset Sources

### PlantVillage
https://www.kaggle.com/datasets/arjuntejaswi/plant-village

### PlantDoc
https://github.com/pratikkayal/PlantDoc-Dataset

---

## Note

The complete dataset is not included in this repository due to size limitations.

Download the datasets from the sources above and organize them according to the structure used in the training notebook before training the model.
