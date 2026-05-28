#!/usr/bin/env python3
"""
Contains all computer vision and image processing functions using the OpenCV library.
"""
import cv2
import numpy as np

def find_blobs_hsv(camera_image: np.ndarray, hsv_ranges: list, min_pixels: int):
    """
    Finds contiguous blobs of pixels matching a given set of HSV color ranges.

    Args:
        camera_image: A BGR numpy array (H x W x 3).
        hsv_ranges: HSV bounds in (H: 0-360, S: 0-100, V: 0-100) format.
        min_pixels: Minimum pixel area for a blob to be returned.
    """
    img_hsv = cv2.cvtColor(camera_image, cv2.COLOR_BGR2HSV_FULL)

    combined_mask = None
    for lower_hsv, upper_hsv in hsv_ranges:
        lower_bound = np.array([lower_hsv[0] * 255/360, lower_hsv[1] * 255/100, lower_hsv[2] * 255/100])
        upper_bound = np.array([upper_hsv[0] * 255/360, upper_hsv[1] * 255/100, upper_hsv[2] * 255/100])
        mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
        combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

    if combined_mask is None:
        return []

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_pixels:
            x, y, w, h = cv2.boundingRect(cnt)
            blobs.append({
                'center_x': x + w / 2,
                'center_y': y + h / 2,
                'height': h,
                'width': w,
                'min_x': x,
                'max_x': x + w,
                'min_y': y,
                'max_y': y + h,
                'area': area,
            })
    return blobs
