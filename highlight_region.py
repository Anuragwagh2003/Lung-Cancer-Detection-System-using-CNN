# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 12:27:01 2025

@author: luiza
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess image (resize, normalize)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (256, 256))  # Resize to model input size
    image_normalized = image_resized / 255.0  # Normalize pixel values
    return image, image_resized, image_normalized

# Function to load the model
def load_segmentation_model(model_path):
    model = load_model(model_path)
    return model

# Function to predict the cancerous region mask
def predict_cancer_region(model, image_normalized):
    input_image = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
    mask = model.predict(input_image)
    return mask[0]  # Return the predicted mask

# Function to highlight cancer regions in the original image
def highlight_regions(original_image, mask):
    # Resize mask to match original image dimensions
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    
    # Threshold the mask (binary mask for cancerous regions)
    _, thresholded_mask = cv2.threshold(mask_resized, 0.5, 1, cv2.THRESH_BINARY)
    
    # Create a highlighted image by coloring the detected regions (green)
    highlighted_region = np.zeros_like(original_image)
    highlighted_region[thresholded_mask == 1] = [0, 255, 0]  # Green color
    
    # Overlay the highlighted regions on the original image
    highlighted_image = cv2.addWeighted(original_image, 0.7, highlighted_region, 0.3, 0)
    
    return highlighted_image

# Main function
if __name__ == "__main__":
    image_path = 'medical_image.jpg'  # Input image file
    model_path = 'unet_model.h5'      # Pretrained U-Net model file
    
    # Preprocess image
    original_image, image_resized, image_normalized = preprocess_image(image_path)
    
    # Load the pretrained model
    model = load_segmentation_model(model_path)
    
    # Predict cancerous regions
    mask = predict_cancer_region(model, image_normalized)
    
    # Highlight the cancerous regions
    highlighted_image = highlight_regions(original_image, mask)
    
    # Display the result
    cv2.imshow('Highlighted Cancer Region', highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
