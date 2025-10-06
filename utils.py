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

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # rotate the kernel by 180 degree
    kernel = np.flipud(np.fliplr(kernel))
    
    for i in range(Hi): 
        for j in range(Wi):
            out[i,j] = np.sum(padded[i:i+Hk, j:j+Wk] * kernel)
    # r, c = Hk//2, Wk//2
    # for di in range(-r, r+1):
    #     for dj in range(-c, c+1):
    #         w = kernel[di + r, dj + c]  
    #         out += w * padded[di + r : di + r + Hi, dj + c : dj + c + Wi]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # for i in range(-size//2 , size//2 + 1):
    #     for j in range(-size//2 , size//2 + 1): 
    #         kernel[i + size//2,j + size//2] = (1/(2*np.pi*sigma**2)) * np.exp(-(i **2 + j **2)/(2*sigma**2))
    
    for i in range(size):
        for j in range(size):
            x = i - size // 2
            y = j - size // 2
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    img = img.astype(np.float32, copy=False)
    kx = np.array([[0.0, 0.0, 0.0],
                   [0.5, 0.0, -0.5],
                   [0.0, 0.0, 0.0]], dtype=np.float32)
    out = conv(img, kx)
    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    img = img.astype(np.float32, copy=False)
    ky = np.array([[0.0,  0.5, 0.0],
                   [0.0,  0.0, 0.0],
                   [0.0, -0.5, 0.0]], dtype=np.float32)
    out = conv(img, ky)
    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    fx = partial_x(img)
    fy = partial_y(img)
    G = np.sqrt(fx**2 + fy**2) 
    theta = np.rad2deg(np.arctan2(fy, fx)) % 360

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    #print(G)
    ### BEGIN YOUR CODE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    theta_q = ((theta % 180.0) + 22.5) // 45.0  # -> {0,1,2,3} sau khi ép int
    theta_q = theta_q.astype(np.int32)
    
    direction = {
        0: (0,1 ) ,
        1: (-1,1) , # top right 
        2: (-1,0) , # top
        3: (-1,-1)  # top left
    }
    
    for i in range(H) :
        for j in range(W) :
            g = G[i,j]
            di, dj = direction[theta_q[i,j]]
            
            n1 = -np.inf
            n2 = -np.inf
            
            i1, j1 = i + di, j + dj
            i2, j2 = i - di, j - dj
            
            if 0 <= i1 < H and 0 <= j1 < W : 
                n1 = G[i1 , j1]
            
            if 0 <= i2  < H and 0 <= j2 < W : 
                n2 = G[i2 , j2]

            out[i,j] = max(  max (g, n1), n2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = (img > high)
    weak_edges = (img > low) & (img <= high)

    return strong_edges, weak_edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    smoothed = conv(img, gaussian_kernel(kernel_size, sigma)) 
    
    G, theta = gradient(smoothed)
    
    nms = non_maximum_suppression(G, theta)
    
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    
    # connect strong edges with weak edges
    H, W = strong_edges.shape

    edges = strong_edges.copy()

    from collections import deque

    q = deque(list(zip(*np.nonzero(strong_edges)))) # list of (y,x) of strong edges
    
    while q : 
        y , x = q.popleft() 
        for ny  in range (y-1 , y+2) :
            for nx in range ( x- 1 , x + 2) :
                if 0 <= ny < H and 0 <= nx < W : 
                    if weak_edges[ny, nx] and not edges[ny, nx] : 
                        edges[ny, nx] = True
                        q.append((ny, nx))
    
   
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return edges
# ================================== load data ======================================

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