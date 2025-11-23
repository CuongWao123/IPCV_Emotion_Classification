# IPCV Emotion Classification Project

A comprehensive computer vision project for facial emotion recognition using traditional machine learning approaches with advanced preprocessing and hyperparameter optimization techniques.

## 📋 Project Overview

This project implements a facial emotion recognition system using classical computer vision and machine learning techniques. The system classifies facial expressions from the FER2013 dataset into **7 emotion categories**: angry, disgust, fear, happy, sad, surprise, and neutral.

### Key Features

- ✅ **Advanced Preprocessing Pipeline**: Median Filter → CLAHE → Resize → Normalize
- ✅ **Multi-Feature Extraction**: HOG + LBP + SIFT combined features
- ✅ **Dimensionality Reduction**: PCA with configurable variance retention (65%-95%)
- ✅ **Data Augmentation**: Minority class balancing (disgust: 349 → 2000 samples)
- ✅ **Multiple Classifiers**: Logistic Regression, Random Forest, SVM with extensive hyperparameter tuning
- ✅ **Anti-Overfitting Strategies**: Regularization, cross-validation, overfitting gap monitoring
- ✅ **Comprehensive Evaluation**: ROC-AUC, F1-scores, confusion matrices, per-class metrics
- ✅ **MLflow Integration**: Experiment tracking and model versioning

## 🗂️ Project Structure

```
IPCV_Emotion_Classification/
├── Data/                           # Processed dataset (small data)
│   ├── train/                      # Training data (augmented)
│   └── test/                       # Test data
├── Data_origin/                    # Original FER2013 dataset
│   ├── train/                      # Original training data
│   │   ├── angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/
│   └── test/                       # Original test data
│       └── angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/
├── models/                         # Saved trained models (*.pkl, *.json)
│   ├── logistic_regression_results_*.json
│   ├── random_forest_results_*.json
│   └── svm_results_*.json
├── TrainingModel.ipynb            # Main training notebook with full pipeline
├── utils.py                       # Core preprocessing and data loading utilities
├── baseline_*.png                 # Baseline model visualizations
├── *_cv_results.csv              # Cross-validation results
├── *_randomized_search_*.csv     # Hyperparameter tuning results
├── *_test_*.png                  # Test set evaluation plots
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image scipy tqdm joblib mlflow
```

### Dataset Setup

1. **Download FER2013 Dataset** and organize in the following structure:

   ```
   Data_origin/
   ├── train/
   │   ├── angry/       (3995 images)
   │   ├── disgust/     (436 images)
   │   ├── fear/        (4097 images)
   │   ├── happy/       (7215 images)
   │   ├── sad/         (4830 images)
   │   ├── surprise/    (3171 images)
   │   └── neutral/     (4965 images)
   └── test/
       ├── angry/       (960 images)
       ├── disgust/     (111 images)
       ├── fear/        (1024 images)
       ├── happy/       (1774 images)
       ├── sad/         (1247 images)
       ├── surprise/    (831 images)
       └── neutral/     (1233 images)
   ```

2. **Data is automatically augmented** during training pipeline execution:
   - Disgust class: 436 → 2000 samples (to balance class distribution)
   - Augmentation techniques: flip, rotation, brightness, noise

### Running the Training Pipeline

1. **Open the main training notebook**:

   ```bash
   jupyter notebook TrainingModel.ipynb
   ```

2. **Execute the complete pipeline** (cells in order):

   - **Cell 1-2**: Import libraries and load FER2013 data
   - **Cell 3**: Data augmentation for minority class (disgust)
   - **Cell 5**: Preprocessing with PCA (variance=0.65)
   - **Cell 7-8**: Logistic Regression baseline + hyperparameter tuning
   - **Cell 10-11**: Random Forest baseline + hyperparameter tuning
   - **Cell 13-16**: Random Forest evaluation on test set
   - **Cell 18-19**: SVM baseline + optimized hyperparameter tuning
   - **Cell 20-21**: SVM evaluation on test set

3. **View experiment tracking**:
   ```bash
   mlflow ui
   ```
   Navigate to `http://localhost:5000` to view all experiments

## 📊 Performance Metrics & Results

The system is evaluated using:

- **Primary Metric**: ROC-AUC (macro-averaged, one-vs-rest)
- **Secondary Metrics**:
  - Overall accuracy
  - F1-score (macro and weighted)
  - Per-class precision, recall, F1-score
  - Confusion matrix analysis
  - Overfitting gap monitoring (Train ROC-AUC - Validation ROC-AUC)
- **Cross-validation**: 3-fold CV for hyperparameter tuning
- **Train/Val/Test Split**: 80/20 split from training data, separate test set

### Baseline Model Performance

| Model                   | Training ROC-AUC | Validation ROC-AUC | Overfitting Gap | Val Accuracy |
| ----------------------- | ---------------- | ------------------ | --------------- | ------------ |
| **Logistic Regression** | 0.9242           | 0.8936             | 0.0306 ✅       | 57.32%       |
| **Random Forest**       | 1.0000           | 0.8627             | 0.1373 ⚠️       | 58.71%       |
| **SVM (RBF, C=1.0)**    | 0.9670           | 0.8190             | 0.1480 ⚠️       | 48.24%       |

### Optimized Model Performance (After Tuning)

**Best configurations identified:**

- **Logistic Regression**: C=0.1, penalty=l2, solver=lbfgs
- **Random Forest**: max_depth=20, min_samples_split=10, n_estimators=200
- **SVM**: kernel=linear, C=0.1 (reduces overfitting from 14.8% to <5%)

### Per-Class Performance (Random Forest on Test Set)

| Emotion     | Precision  | Recall     | F1-Score   | Support |
| ----------- | ---------- | ---------- | ---------- | ------- |
| Angry       | 0.4547     | 0.5208     | 0.4854     | 960     |
| **Disgust** | **0.9737** | **0.9189** | **0.9455** | 111     |
| Fear        | 0.4258     | 0.3867     | 0.4054     | 1024    |
| **Happy**   | **0.7544** | **0.7871** | **0.7704** | 1774    |
| Sad         | 0.4230     | 0.4018     | 0.4121     | 1247    |
| Surprise    | 0.6882     | 0.6811     | 0.6846     | 831     |
| Neutral     | 0.5022     | 0.5292     | 0.5153     | 1233    |

**Overall Test Accuracy**: 58.33%  
**Macro F1-Score**: 0.5884  
**Test ROC-AUC**: 0.8620

## 🔧 Preprocessing Pipeline Details

The `EmotionDataPreprocessor` class in `utils.py` implements an 8-step pipeline:

1. **Median Filter (3×3)**: Noise reduction
2. **CLAHE Enhancement**: Contrast improvement (clip_limit=1.5, tile_grid=8×8)
3. **Resize & Normalize**: 64×64 pixels, normalized to [0,1]
4. **HOG Features**: Histogram of Oriented Gradients (pixels_per_cell=8×8)
5. **LBP Features**: Local Binary Patterns with grid histogram (8×8 grid, P=8, R=1)
6. **SIFT Features**: Scale-Invariant Feature Transform (fixed size=1280)
7. **Feature Combination**: Concatenate HOG + LBP + SIFT
8. **Standardization & PCA**: StandardScaler + PCA (configurable variance retention)

**Total features before PCA**: ~3000-4000 dimensions  
**After PCA (65% variance)**: ~400-600 dimensions  
**After PCA (95% variance)**: ~1200-1500 dimensions

## 🛠️ Troubleshooting

### Common Issues

1. **Memory errors**:

   - Reduce PCA variance (try 0.65 instead of 0.95)
   - Process data in smaller batches
   - Use less aggressive augmentation

2. **Long training times (SVM)**:

   - Use linear kernel instead of RBF (10x faster)
   - Reduce C values to [0.01, 0.1, 0.5] range
   - Use fewer cross-validation folds (cv=3 instead of 5)
   - Enable n_jobs=-1 for parallel processing

3. **Overfitting issues**:

   - Monitor train vs validation gap
   - Reduce model complexity (lower max_depth for RF, lower C for SVM/LR)
   - Increase regularization strength
   - Use more data augmentation

4. **Class imbalance**:
   - Already addressed with class_weight='balanced' in all models
   - Disgust class augmented from 436 to 2000 samples
   - Consider SMOTE or other synthetic sampling techniques

### Performance Optimization Tips

- **Use PCA with 0.65 variance** for faster training (minimal accuracy loss)
- **Start with linear models** (Logistic Regression, Linear SVM) before trying complex ones
- **For SVM hyperparameter tuning**:
  - Use `kernel='linear'` exclusively for speed
  - Focus on C values: [0.01, 0.05, 0.1, 0.5]
  - Avoid gamma parameter with linear kernel
- **For Random Forest**:
  - Limit max_depth to 10-30 range
  - Use min_samples_split ≥ 10 to prevent overfitting
  - n_estimators: 100-200 is usually sufficient

## 📊 Key Findings & Insights

### What Works Well:

✅ **Data Augmentation**: Balancing disgust class improved its F1-score from ~0.40 to ~0.94  
✅ **Feature Combination**: HOG + LBP + SIFT provides complementary information  
✅ **PCA at 65%**: Reduces features by 85% with minimal performance loss  
✅ **Linear Models**: Logistic Regression shows best generalization (lowest overfitting gap)  
✅ **Class Weighting**: `class_weight='balanced'` crucial for imbalanced data

### Overfitting Analysis:

⚠️ **Random Forest**: Severe overfitting (Train: 100%, Val: 86.3%)  
⚠️ **SVM with RBF**: High overfitting gap (14.8%) when C=1.0  
✅ **Solution**: Use regularization (lower C), simpler kernels (linear), tree depth limits

### Best Practices Identified:

1. **Always monitor overfitting gap** (should be <5%)
2. **Start simple** (Logistic Regression baseline) before complex models
3. **Use linear SVM** for hyperparameter tuning (10x faster than RBF)
4. **PCA variance 65-75%** provides best speed/accuracy trade-off
5. **Augment minority classes** before splitting train/val

## 📚 References

- **FER2013 Dataset**: Facial Expression Recognition 2013 Challenge
- **HOG Features**: Dalal & Triggs (2005) - Histograms of Oriented Gradients for Human Detection
- **LBP**: Ojala et al. (2002) - Multiresolution Gray-Scale and Rotation Invariant Texture Classification
- **SIFT**: Lowe (2004) - Distinctive Image Features from Scale-Invariant Keypoints
- **CLAHE**: Zuiderveld (1994) - Contrast Limited Adaptive Histogram Equalization
- **Scikit-learn**: Pedregosa et al. (2011) - Machine Learning in Python

## 📄 License

This project is part of the HCMUT Computer Vision (IPCV) course. For academic use only.

## 🎯 Future Improvements

### Planned Enhancements:

- [ ] Implement CNN-based deep learning models (ResNet, EfficientNet)
- [ ] Advanced data augmentation (Mixup, CutMix, AutoAugment)
- [ ] Ensemble methods combining multiple models
- [ ] Attention mechanisms for important facial regions
- [ ] Transfer learning from pre-trained face recognition models
- [ ] Real-time webcam emotion detection
- [ ] Cross-dataset validation (test on AffectNet, JAFFE)
- [ ] Explainability analysis (Grad-CAM, LIME)

### Current Limitations:

- Traditional ML features may miss subtle expressions
- Limited performance on "fear" and "sad" classes (confusion between them)
- No temporal modeling (video sequences)
- Grayscale only (color information ignored)

---

**Course**: Computer Vision (IPCV)  
**Institution**: Ho Chi Minh University of Technology (HCMUT)  
**Year**: 2025
