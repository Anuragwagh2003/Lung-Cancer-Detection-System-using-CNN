# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 18:11:42 2025

@author: luiza
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def highlight_cancerous_region(image_path: str, mask: np.ndarray) -> np.ndarray:
    """
    Highlights cancerous regions in the image based on the provided mask.

    Parameters:
    - image_path: str, path to the input image.
    - mask: np.ndarray, binary mask where non-zero pixels represent cancerous regions.

    Returns:
    - np.ndarray: Image with highlighted cancerous regions.
    """
    # Load the original image
    image = cv2.imread(image_path)

    # Ensure the mask is in the same size as the image
    if image.shape[:2] != mask.shape:
        raise ValueError("Mask and image dimensions do not match.")

    # Convert the mask to 3 channels
    mask_colored = cv2.merge([mask, mask, mask])

    # Highlight the cancerous regions by blending the mask with the original image
    highlighted_image = cv2.addWeighted(image, 0.7, mask_colored.astype(np.uint8), 0.3, 0)

    return highlighted_image

def display_image(image: np.ndarray, title: str = "Highlighted Image") -> None:
    """
    Displays the image using matplotlib.

    Parameters:
    - image: np.ndarray, the image to display.
    - title: str, the title of the displayed image.
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
