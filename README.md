# 🏥 Automated Kidney Stone Detection Using CNN and YOLOv5

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![Published](https://img.shields.io/badge/Published-Grenze%20Journal-success)](https://grenzejournal.com)

> **Published Research** | Grenze International Journal of Engineering and Technology | June 2025

---

## 📄 Publication Details

**Title:** Automated Stone Detection using CNN and YOLOv5 Approach with Deep Learning Techniques

**Authors:** Divyadharshini J, **Yogesh Kumar S**, Srinidhi S, Manikandan K

**Institution:** Vellore Institute of Technology, Tamil Nadu, India

**Journal:** Grenze International Journal of Engineering and Technology (GIJET)

**Publication Date:** June 2025

**DOI:** 01.GIJET.11.2.228_25

---

## 🔬 Research Overview

This project presents an **automated deep learning-based system** for kidney stone detection and classification using hybrid CNN-YOLOv5 architecture. The system addresses critical challenges in medical imaging by providing:

- ✅ Real-time kidney stone detection in CT/MRI scans
- ✅ Multi-class classification (Normal Kidney, Kidney Stones, Background)
- ✅ Automated diagnosis with minimal human intervention
- ✅ Enhanced interpretability using Grad-CAM visualization

### Key Contributions

1. **Hybrid Architecture**: Combined CNN feature extraction with YOLOv5 object detection
2. **High Accuracy**: Achieved 90%+ F1-score and 82.7% Average Precision
3. **Clinical Applicability**: Real-time processing suitable for hospital workflows
4. **Explainable AI**: Integrated Grad-CAM for model interpretability

---

## 🛠️ Technical Architecture

### System Pipeline
```
CT/MRI Scan Input
    ↓
Image Preprocessing (OpenCV)
    ↓
CNN Feature Extraction
    ↓
YOLOv5 Object Detection & Localization
    ↓
Grad-CAM Visualization
    ↓
Classification Output + Bounding Boxes
```

### Technologies Used

- **Deep Learning Frameworks:** PyTorch, YOLOv5
- **Computer Vision:** OpenCV-Python
- **Model Interpretability:** Grad-CAM
- **Data Augmentation:** SMOTE (Synthetic Minority Over-sampling)
- **Languages:** Python 3.8+

---

## 📊 Performance Metrics

### Classification Results

| Metric | Value |
|--------|-------|
| **F1-Score (Max)** | 0.90 @ confidence 0.414 |
| **Precision** | 0.883 @ high confidence |
| **Recall** | 0.90 |
| **mAP@0.5** | 0.90 |
| **mAP@0.5:0.95** | 0.65 |

### Class-wise Performance

| Class | Average Precision (AP) | Accuracy |
|-------|----------------------|----------|
| Normal Kidney | 0.995 | 100% |
| Kidney Stone (Tas_Var) | 0.827 | 84% |
| Background | N/A | High |

### Training Metrics

- **Epochs:** 40
- **Box Loss:** Reduced from 0.10 → 0.04 (60% improvement)
- **Objectness Loss:** 0.018 → 0.008 (55% improvement)
- **Classification Loss:** 0.015 → ~0.000 (near perfect)

---

## 🎯 Model Architecture Details

### YOLOv5 Configuration
- **Version:** YOLOv5s/YOLOv5m (configurable)
- **Input Size:** Resized and normalized CT/MRI images
- **Data Split:** 70% Training / 30% Testing
- **Optimization:** Non-maximum suppression for duplicate removal

### CNN Features
- **Feature Extraction:** Spatial hierarchies for pattern recognition
- **Classification:** Multi-class output (Normal/Stone/Background)
- **Augmentation:** Flipping, rotation, SMOTE balancing

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 1.9+
CUDA (optional, for GPU acceleration)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/kidney-stone-detection.git
cd kidney-stone-detection

# Install dependencies
pip install -r requirements.txt

# Download YOLOv5 weights (if not included)
# Place weights in models/ directory
```

### Expected Output

- Bounding boxes around detected kidney stones
- Confidence scores for each detection
- Class labels (Normal Kidney / Kidney Stone)
- Grad-CAM heatmap overlay (if enabled)

---

## 📁 Project Structure
```
kidney-stone-detection/
├── models/
│   ├── yolov5_model.py      # YOLOv5 architecture
│   ├── cnn_model.py          # CNN classifier
│   └── best.pt               # Trained weights
├── utils/
│   ├── preprocessing.py      # Image preprocessing (OpenCV)
│   ├── gradcam.py            # Grad-CAM implementation
│   └── visualization.py      # Result visualization
├── data/
│   ├── train/                # Training images
│   ├── test/                 # Testing images
│   └── annotations/          # Bounding box labels
├── detect.py                 # Inference script
├── train.py                  # Training script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 🔬 Research Methodology

### Data Collection & Preprocessing
1. **Dataset:** CT and MRI scans of kidney images
2. **Annotation:** Bounding boxes using LabelImg tool
3. **Preprocessing:** OpenCV-Python for resizing and normalization
4. **Balancing:** SMOTE technique for class distribution

### Model Training
1. **Architecture Selection:** YOLOv5s/m based on task complexity
2. **Loss Functions:** Object localization + classification loss
3. **Optimization:** Batch size and learning rate tuning
4. **Validation:** 30% holdout set for generalization testing

### Evaluation
- Confusion matrix analysis
- Precision-Recall curves
- F1-Confidence curves
- mAP metrics at different IoU thresholds

---

## 📈 Key Results & Findings

### Confusion Matrix Insights
- **Normal Kidney:** 100% classification accuracy (distinctive features)
- **Kidney Stones:** 84% accuracy with 16% background misclassification
- **Feature Overlap:** Some confusion between stones and background regions

### Optimal Operating Point
- **Best F1-Score:** 0.90 at confidence threshold 0.414
- **Precision-Recall Trade-off:** High precision (0.883) with strong recall (0.90)
- **Clinical Recommendation:** Use 0.414 threshold for balanced performance

### Model Robustness
- Consistent performance across training and validation sets
- Low overfitting (training and validation losses converge)
- Good generalization to unseen medical images

---

## 🔮 Future Enhancements

- [ ] **Multi-class Stone Types:** Classify calcium oxalate, uric acid, struvite, etc.
- [ ] **Real-time Video Processing:** Live ultrasound stream analysis
- [ ] **Mobile Deployment:** TensorFlow Lite for edge devices
- [ ] **Clinical Integration:** PACS system compatibility
- [ ] **Federated Learning:** Privacy-preserving multi-hospital training
- [ ] **Web Interface:** User-friendly diagnostic platform
- [ ] **Larger Dataset:** Expand to 10,000+ annotated scans

---

## 📚 Citation

If you use this work in your research, please cite:
```bibtex
@article{divyadharshini2025kidney,
  title={Automated Stone Detection using CNN and YOLOv5 Approach with Deep Learning Techniques},
  author={Divyadharshini, J and Kumar, Yogesh S and Srinidhi, S and Manikandan, K},
  journal={Grenze International Journal of Engineering and Technology},
  volume={11},
  number={2},
  pages={8870--8875},
  year={2025},
  publisher={Grenze Scientific Society},
  doi={01.GIJET.11.2.228_25}
}
```

---

## 🏆 Achievements

- ✅ Published in **peer-reviewed journal** (Grenze GIJET)
- ✅ Presented at **VIT Research Symposium**
- ✅ **90% F1-Score** on real-world medical imaging data
- ✅ Contributions to **automated medical diagnostics** field

---

## 👥 Team

**Research Team:**
- **Divyadharshini J** - Conceptualization, Implementation
- **Yogesh Kumar S** - Coding, Implementation, Data Analysis
- **Srinidhi S** - Literature Survey, Data Analysis
- **Dr. Manikandan K** - Supervision, Manuscript Review

**Institution:** School of Computer Science and Engineering, Vellore Institute of Technology

---

## 📧 Contact

**Yogesh Kumar S**  
AI/ML Engineer | Published Researcher | M.Tech CSE @ VIT

📧 mail.syogeshk@gmail.com  

---

## 🙏 Acknowledgments

- Vellore Institute of Technology for research support and infrastructure
- Medical imaging dataset contributors
- Open-source community (PyTorch, YOLOv5, OpenCV)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Star This Repository!

If you find this research useful, please ⭐ star this repository and share it with others working in medical imaging and deep learning!

---

**Keywords:** Kidney Stone Detection, Deep Learning, CNN, YOLOv5, Medical Imaging, Computer Vision, PyTorch, Grad-CAM, Automated Diagnosis, Medical AI, Healthcare Technology
