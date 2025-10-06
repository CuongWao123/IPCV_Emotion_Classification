import os
import shutil
import random
from pathlib import Path

def extract_sample_images():
    """
    Extract 30 random images from each emotion category in the train folder
    and copy them to the Data/sample/ folder with the same emotion structure.
    """
    
    # Define paths
    train_path = Path("Data/train")
    sample_path = Path("Data/sample")
    
    # List of emotion categories
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    # Number of images to extract per emotion
    images_per_emotion = 30
    
    print("Starting to extract sample images...")
    print(f"Target: {images_per_emotion} images per emotion category")
    print("-" * 50)
    
    for emotion in emotions:
        print(f"\nProcessing emotion: {emotion}")
        
        # Source and destination paths
        source_dir = train_path / emotion
        dest_dir = sample_path / emotion
        
        # Get all image files from the source directory
        image_files = list(source_dir.glob("*.jpg"))
        
        print(f"Found {len(image_files)} images in {source_dir}")
        
        if len(image_files) < images_per_emotion:
            print(f"Warning: Only {len(image_files)} images available, copying all of them.")
            selected_images = image_files
        else:
            # Randomly select 30 images
            selected_images = random.sample(image_files, images_per_emotion)
        
        # Copy selected images to destination
        copied_count = 0
        for img_file in selected_images:
            try:
                dest_file = dest_dir / img_file.name
                shutil.copy2(img_file, dest_file)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {img_file.name}: {e}")
        
        print(f"Successfully copied {copied_count} images to {dest_dir}")
    
    print("\n" + "=" * 50)
    print("Sample image extraction completed!")
    
    # Verify the results
    print("\nVerification:")
    for emotion in emotions:
        dest_dir = sample_path / emotion
        count = len(list(dest_dir.glob("*.jpg")))
        print(f"{emotion}: {count} images")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run the extraction
    extract_sample_images()