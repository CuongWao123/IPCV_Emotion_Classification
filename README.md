# IPCV Emotion Classification Project

A comprehensive computer vision project for facial emotion recognition using traditional machine learning approaches and advanced preprocessing techniques.

## 📋 Project Overview

This project implements a facial emotion recognition system using classical computer vision and machine learning techniques. The system classifies facial expressions into 7 emotion categories: **angry**, **disgust**, **fear**, **happy**, **neutral**, **sad**, and **surprise**.

### Key Features

- **Preprocessing pipeline** Median filter , CLAHE , Resize , Normalize
- **Feature extraction** HOG , LBP , SIFT
- **Multiple classifier comparison** (SVM, Random Forest, Logistic Regression, XGBoost)
- **Comprehensive evaluation** with visualization and analysis tools

## 🗂️ Project Structure

```
IPCV_Project/
├── Data/
│   ├── train/          # Training dataset (FER2013 format)
│   ├── test/           # Testing dataset  
│   └── sample/         # Sample images for quick testing
├── models/
│   └── *.pkl    # Trained models
├── Experiments.ipynb   # Main experimental notebook
├── utils.py           # Utility functions and preprocessing
├── extract_sample_images.py  # Data sampling script
└── README.md          # This file
```

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install numpy matplotlib opencv-python scikit-learn scikit-image tensorflow keras pandas seaborn tqdm mtcnn lz4 facenet_pytorch xgboost
```

### Dataset Setup

1. **FER2013 Dataset**: Organize your data in the following structure:
   ```
   Data/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   └── test/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── neutral/
       ├── sad/
       └── surprise/
   ```

2. **Extract Sample Data** (optional for quick testing):
   ```bash
   python extract_sample_images.py
   ```
   This creates a sample dataset with 30 images per emotion for rapid prototyping.

### Running the Experiments

1. **Open the main notebook**:
   ```bash
   jupyter notebook Experiments.ipynb
   ```

2. **Run the complete pipeline**:
   - Data loading and preprocessing
   - Feature extraction and dimensionality reduction
   - Model training and evaluation
   - Results visualization and analysis

## 📊 Performance Metrics

The system is evaluated using:

- **Primary Metric**: Macro-averaged F1-score
- **Secondary Metrics**: 
  - Overall accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrix analysis
- **Cross-validation**: Stratified 5-fold CV for model selection

### Expected Performance

| Emotion | Typical F1-Score | Notes |
|---------|------------------|-------|
| Happy | 0.75-0.85 | Best performing class |
| Angry | 0.65-0.75 | Good discrimination |
| Surprise | 0.70-0.80 | Clear facial features |
| Neutral | 0.60-0.70 | Moderate difficulty |
| Sad | 0.55-0.70 | Often confused with angry |
| Fear | 0.50-0.65 | Confused with surprise |
| Disgust | 0.40-0.60 | Most challenging class |


## 🛠️ Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use data generators
2. **MTCNN failures**: Check image quality and face visibility
3. **Poor performance**: Verify data preprocessing pipeline
4. **Missing dependencies**: Install all required packages

### Performance Optimization

- Use PCA for dimensionality reduction
- Apply class weighting for imbalanced data
- Consider ensemble methods for improved accuracy
- Implement early stopping for iterative models

## 📚 References

- **FER2013 Dataset**: Facial Expression Recognition 2013 Challenge
- **HOG Features**: Dalal & Triggs (2005) - Histograms of Oriented Gradients
- **MTCNN**: Zhang et al. (2016) - Joint Face Detection and Alignment  
- **CLAHE**: Zuiderveld (1994) - Contrast Limited Adaptive Histogram Equalization

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is part of the HCMUT Computer Vision course. Please follow academic integrity guidelines.

## 🎯 Future Improvements

- [ ] Deep learning models (CNN architectures)
- [ ] Data augmentation techniques
- [ ] Real-time emotion recognition
- [ ] Multi-modal fusion (audio + visual)
- [ ] Transfer learning from pre-trained models
- [ ] Advanced ensemble methods

---

**Course**: Computer Vision (IPCV)  
**Institution**: Ho Chi Minh University of Technology (HCMUT)  
**Year**: 2025