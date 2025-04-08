# Scalable Aerial GNSS Localization for Marine Robots

**Authors:** Shuo Wen*, Edwin Meriaux*, Mariana Sosa Guzmán, Charlotte Morissette, Chuqiao Si, Bobak Baghi, Gregory Dudek  
*\*Co-first authors*  

This repository contains the implementation of our drone-based framework for **localizing surfaced marine robots using aerial GNSS-equipped drones and vision-based estimation**. This scalable and cost-efficient method supports both single and multi-robot tracking using YOLOv11 and geometric inference.

---

## 🌊 Abstract
Accurate localization is crucial for underwater robotics, yet traditional onboard Global Navigation Satellite System (GNSS) approaches are difficult or ineffective due to signal reflection on the water’s surface. Existing approaches, such as inertial navigation, Doppler Velocity Loggers (DVL), SLAM and acoustic-based methods, face challenges like error accumulation and high computational complexity. Therefore, a more efficient and scalable solution remains necessary. This paper proposes an alternative approach that leverages an aerial drone equipped with GNSS localization to track and localize an underwater robot near the surface. Our results show that this novel adaptation enables accurate single and multi-robot underwater localization.  

## 🚀 Overview

Traditional underwater localization methods suffer from signal loss, sensor drift, or infrastructure cost. We propose an aerial solution using:

- A GNSS-enabled drone
- Vision-based detection (YOLOv11)
- Geometric estimation

This project demonstrates real-time, accurate GNSS localization of marine robots (e.g., Aqua2) during surfacing events.

---

## 🧭 Repository Structure

```
.
├── clean_data.py              # Remove invalid image/label pairs (out-of-bounds boxes)
├── image_augmentation.py      # Applies augmentations (blur, fog, flips, etc.)
├── estimator_func.py          # Core logic for GNSS triangulation
├── yolo11n_trained.pt         # Trained YOLOv11 for single robot
├── yolo11n_multiaqua.pt       # Trained YOLOv11 for multi-robot
├── single.csv                 # Single robot tracking predictions + error
├── multi.csv                  # Multi-robot tracking predictions + error
├── requirements.txt           # Dependencies
```

---


> See `image_augmentation.py` and `clean_data.py` for image augmentation pipeline and filtering.


---

## ⚙️ Setup Instructions

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run augmentation:

```bash
python image_augmentation.py
```

Clean invalid samples (if needed):

```bash
python clean_data.py
```

To see our demo, please run

For single Aqua
```bash
python single_aqua.py
```

For multi Aqua
```bash
python single_aqua.py
```



---

## 📚 Citation

If this project helps your research, please cite:

```bibtex
@inproceedings{wen2025scalable,
  title={Scalable Aerial GNSS Localization for Marine Robots},
  author={Wen, Shuo and Meriaux, Edwin and Sosa Guzm{\'a}n, Mariana and Morissette, Charlotte and Si, Chuqiao and Baghi, Bobak and Dudek, Gregory},
  booktitle={ICRA Workshop on Marine Robotics},
  year={2025}
}
```

---

## 📨 Contact

For questions or collaboration, contact:

- Shuo Wen & Edwin Meriaux — McGill CIM Lab | Email: shuo.wen@mail.mcgill.ca edwin.meriaux@mail.mcgill.ca
- Project repo: https://github.com/stevvwen/aerial_gnss
