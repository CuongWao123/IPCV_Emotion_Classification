import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm 
import cv2 
import os
from skimage.feature import hog, local_binary_pattern
import joblib 


def load_fer2013_data(data_path, target_size=(48, 48)):
    """
    Load FER2013 dataset and preprocess images
    """
    images = []
    labels = []
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    print("Loading and preprocessing images...")
    
    for emotion_idx, emotion in enumerate(emotion_labels):
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"Warning: {emotion_path} not found")
            continue
            
        image_files = os.listdir(emotion_path)
        print(f"Processing {emotion}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"Loading {emotion}"):
            img_path = os.path.join(emotion_path, img_file)
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Convert to grayscale
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to target size
            img = cv2.resize(img, target_size)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(emotion_idx)
    
    return np.array(images), np.array(labels), emotion_labels


class EmotionDataPreprocessor:
    """
    Complete preprocessing pipeline for emotion classification
    Includes: Median Filter, CLAHE, HOG, LBP, SIFT, Scaling, PCA
    """
    
    def __init__(self, target_size=(64, 64), random_state=42):
        """
        Initialize preprocessor
        
        Parameters:
        - target_size: Target image size (height, width)
        - random_state: Random seed for reproducibility
        """
        self.target_size = target_size
        self.random_state = random_state
        
        # Components to be fitted
        self.scaler = StandardScaler()
        self.pca = None
        
        # CLAHE parameters
        self.clahe_clip_limit = 2.0
        self.clahe_tile_grid = (8, 8)
        
        # HOG parameters
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
        
        # LBP parameters
        self.lbp_P = 8
        self.lbp_R = 1
        self.lbp_grid = (8, 8)
        
        # SIFT parameters
        self.sift_fixed_size = 128 * 10  # 1280 features
        
        # Fitted flag
        self.is_fitted = False
        
    
    def _apply_median_filter(self, images):
        """Apply median filter to reduce noise"""
        print("  → Applying Median Filter...")
        return np.array([cv2.medianBlur(img, 3) for img in tqdm(images, desc="Median Filter")])
    
    
    def _apply_clahe(self, images):
        """Apply CLAHE for contrast enhancement"""
        print("  → Applying CLAHE Enhancement...")
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
        enhanced_images = []
        
        for img in tqdm(images, desc="CLAHE"):
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img_uint8 = (img * 255).astype(np.uint8)
                else:
                    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img_uint8 = img
            
            img_enhanced = clahe.apply(img_uint8)
            enhanced_images.append(img_enhanced)
        
        return np.array(enhanced_images)
    
    
    def _resize_and_normalize(self, images):
        """Resize images and normalize to [0, 1]"""
        print(f"  → Resizing to {self.target_size} and Normalizing...")
        resized = np.array([cv2.resize(img, self.target_size) for img in tqdm(images, desc="Resize")])
        normalized = resized.astype('float32') / 255.0
        with_channel = normalized.reshape(-1, self.target_size[0], self.target_size[1], 1)
        return with_channel
    

    def _extract_hog_features(self, images):
        """Extract HOG features"""
        print("  → Extracting HOG Features...")
        hog_features = []

        for img in tqdm(images, desc="HOG"):
            features = hog(
                img.squeeze(),
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                visualize=False
            )
            hog_features.append(features)
        
        result = np.array(hog_features)
        print(f"    HOG shape: {result.shape}")
        return result
    
    
    def _lbp_grid_hist(self, img):
        """Compute LBP grid histogram for a single image"""
        H, W = img.shape
        gy, gx = self.lbp_grid
        lbp = local_binary_pattern(img, P=self.lbp_P, R=self.lbp_R, method='uniform')
        
        n_bins = self.lbp_P + 2
        cell_h, cell_w = H // gy, W // gx

        feats = []
        for i in range(gy):
            for j in range(gx):
                y0, y1 = i*cell_h, (i+1)*cell_h if i<gy-1 else H
                x0, x1 = j*cell_w, (j+1)*cell_w if j<gx-1 else W
                cell = lbp[y0:y1, x0:x1].ravel()
                hist, _ = np.histogram(cell, bins=np.arange(0, n_bins+1), range=(0, n_bins))
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-8)
                feats.append(hist)
        
        return np.concatenate(feats, axis=0)
    
    
    def _extract_lbp_features(self, images):
        """Extract LBP features"""
        print("  → Extracting LBP Features...")
        lbp_features = []
        
        for img in tqdm(images, desc="LBP"):
            features = self._lbp_grid_hist(img.squeeze())
            lbp_features.append(features)
        
        result = np.array(lbp_features)
        print(f"    LBP shape: {result.shape}")
        return result
    
    
    def _extract_sift_features(self, images):
        """Extract SIFT features"""
        print("  → Extracting SIFT Features...")
        sift_features = []
        sift = cv2.SIFT_create()
        
        for img in tqdm(images, desc="SIFT"):
            keypoints, descriptors = sift.detectAndCompute(
                (img.squeeze() * 255).astype('uint8'), None
            )
            
            if descriptors is not None:
                sift_feature = descriptors.flatten()
                if sift_feature.shape[0] < self.sift_fixed_size:
                    # Pad to fixed size
                    sift_feature = np.pad(
                        sift_feature, 
                        (0, self.sift_fixed_size - sift_feature.shape[0]), 
                        'constant'
                    )
                else:
                    # Truncate if too long
                    sift_feature = sift_feature[:self.sift_fixed_size]
            else:
                # No keypoints found
                sift_feature = np.zeros(self.sift_fixed_size)
            
            sift_features.append(sift_feature)
        
        result = np.array(sift_features)
        print(f"    SIFT shape: {result.shape}")
        return result
    
    
    def fit_transform(self, X_raw, pca_variance=0.95):
        """
        Fit and transform training data through complete pipeline
        
        Parameters:
        - X_raw: Raw training images
        - pca_variance: Variance to retain in PCA (default: 0.95)
        
        Returns:
        - X_processed: Processed features ready for training
        - preprocessing_info: Dictionary with intermediate results
        """
        print("\n" + "="*70)
        print(" TRAINING DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        preprocessing_info = {}
        
        # Step 1: Median Filter
        print("\n[1/8] Median Filter")
        X_median = self._apply_median_filter(X_raw)
        preprocessing_info['median_filtered'] = X_median
        
        # Step 2: CLAHE Enhancement
        print("\n[2/8] CLAHE Enhancement")
        X_clahe = self._apply_clahe(X_median)
        preprocessing_info['clahe_enhanced'] = X_clahe
        
        # Step 3: Resize and Normalize
        print("\n[3/8] Resize and Normalize")
        X_normalized = self._resize_and_normalize(X_clahe)
        preprocessing_info['normalized'] = X_normalized
        
        # Step 4: HOG Feature Extraction
        print("\n[4/8] HOG Feature Extraction")
        X_hog = self._extract_hog_features(X_normalized)
        preprocessing_info['hog_features'] = X_hog
        
        # Step 5: LBP Feature Extraction
        print("\n[5/8] LBP Feature Extraction")
        X_lbp = self._extract_lbp_features(X_normalized)
        preprocessing_info['lbp_features'] = X_lbp
        
        # Step 6: SIFT Feature Extraction
        print("\n[6/8] SIFT Feature Extraction")
        X_sift = self._extract_sift_features(X_normalized)
        preprocessing_info['sift_features'] = X_sift
        
        # Step 7: Combine Features
        print("\n[7/8] Combining Features (HOG + LBP + SIFT)")
        X_combined = np.concatenate([X_hog, X_lbp, X_sift], axis=1)
        print(f"  → Combined shape: {X_combined.shape}")
        print(f"    - HOG: {X_hog.shape[1]} features")
        print(f"    - LBP: {X_lbp.shape[1]} features")
        print(f"    - SIFT: {X_sift.shape[1]} features")
        print(f"    - Total: {X_combined.shape[1]} features")
        preprocessing_info['combined'] = X_combined
        
        # Step 8: Standard Scaling and PCA
        print("\n[8/8] Standard Scaling and PCA")
        print("  → Fitting StandardScaler...")
        X_scaled = self.scaler.fit_transform(X_combined)
        preprocessing_info['scaled'] = X_scaled
        
        print(f"  → Fitting PCA (variance={pca_variance})...")
        self.pca = PCA(n_components=pca_variance, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        preprocessing_info['pca'] = X_pca
        
        print(f"    PCA reduced: {X_combined.shape[1]} → {X_pca.shape[1]} features")
        print(f"    Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        print(f"    Number of components: {self.pca.n_components_}")
        
        self.is_fitted = True
        
        print("\n" + "="*70)
        print(" TRAINING PREPROCESSING COMPLETED")
        print("="*70)
        print(f"Final shape: {X_pca.shape}")
        print(f"Original samples: {len(X_raw)}")
        
        return X_pca, preprocessing_info
    
    
    def transform(self, X_raw):
        """
        Transform test data using fitted pipeline
        
        Parameters:
        - X_raw: Raw test images
        
        Returns:
        - X_processed: Processed features ready for prediction
        - preprocessing_info: Dictionary with intermediate results
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first! Call fit_transform() on training data.")
        
        print("\n" + "="*70)
        print(" TEST DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        preprocessing_info = {}
        
        # Step 1: Median Filter
        print("\n[1/8] Median Filter")
        X_median = self._apply_median_filter(X_raw)
        preprocessing_info['median_filtered'] = X_median
        
        # Step 2: CLAHE Enhancement
        print("\n[2/8] CLAHE Enhancement")
        X_clahe = self._apply_clahe(X_median)
        preprocessing_info['clahe_enhanced'] = X_clahe
        
        # Step 3: Resize and Normalize
        print("\n[3/8] Resize and Normalize")
        X_normalized = self._resize_and_normalize(X_clahe)
        preprocessing_info['normalized'] = X_normalized
        
        # Step 4: HOG Feature Extraction
        print("\n[4/8] HOG Feature Extraction")
        X_hog = self._extract_hog_features(X_normalized)
        preprocessing_info['hog_features'] = X_hog
        
        # Step 5: LBP Feature Extraction
        print("\n[5/8] LBP Feature Extraction")
        X_lbp = self._extract_lbp_features(X_normalized)
        preprocessing_info['lbp_features'] = X_lbp
        
        # Step 6: SIFT Feature Extraction
        print("\n[6/8] SIFT Feature Extraction")
        X_sift = self._extract_sift_features(X_normalized)
        preprocessing_info['sift_features'] = X_sift
        
        # Step 7: Combine Features
        print("\n[7/8] Combining Features (HOG + LBP + SIFT)")
        X_combined = np.concatenate([X_hog, X_lbp, X_sift], axis=1)
        print(f"  → Combined shape: {X_combined.shape}")
        preprocessing_info['combined'] = X_combined
        
        # Step 8: Standard Scaling and PCA (using fitted transformers)
        print("\n[8/8] Standard Scaling and PCA")
        print("  → Applying fitted StandardScaler...")
        X_scaled = self.scaler.transform(X_combined)  # Use transform, not fit_transform
        preprocessing_info['scaled'] = X_scaled
        
        print("  → Applying fitted PCA...")
        X_pca = self.pca.transform(X_scaled)  # Use transform, not fit_transform
        preprocessing_info['pca'] = X_pca
        
        print("\n" + "="*70)
        print(" TEST PREPROCESSING COMPLETED")
        print("="*70)
        print(f"Final shape: {X_pca.shape}")
        print(f"Original samples: {len(X_raw)}")
        
        return X_pca, preprocessing_info
    
    
    def get_params(self):
        """Get current preprocessing parameters"""
        return {
            'target_size': self.target_size,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_grid': self.clahe_tile_grid,
            'hog_pixels_per_cell': self.hog_pixels_per_cell,
            'hog_cells_per_block': self.hog_cells_per_block,
            'lbp_P': self.lbp_P,
            'lbp_R': self.lbp_R,
            'lbp_grid': self.lbp_grid,
            'sift_fixed_size': self.sift_fixed_size,
            'pca_components': self.pca.n_components_ if self.is_fitted else None,
            'is_fitted': self.is_fitted
        }
    
    
    def save(self, filepath):
        """Save preprocessor to file"""
        import joblib
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to: {filepath}")
    
    
    @staticmethod
    def load(filepath):
        """Load preprocessor from file"""
        import joblib
        preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from: {filepath}")
        return preprocessor

