# -*- coding:utf8 -*-
# !/usr/bin/env python3
# Copyright 2025 Volodymyr Pavliukevych
# Author Volodymyr Pavliukevych

"""This is a POC page representation
"""
import os 
from PIL import Image, ImageCms

import cv2
import numpy as np
from uuid import uuid4

def read_and_resize_large_image(src_image_path: str, dst_image_path:str, shape:tuple=None, ratio:float=None) -> tuple:
    """
    Reads a large image (TIFF or PNG) and downsizes it to 3000x3000 pixels.

    Parameters:
        image_path (str): Path to the large image file.

    Returns:
        numpy.ndarray: Resized image of size shape 3000x3000 pixels.
    """
    # Read the image using OpenCV
    large_image = cv2.imread(src_image_path, cv2.IMREAD_UNCHANGED)

    if large_image is None:
        raise ValueError(f"Could not read the image from path: {src_image_path}")
    elif large_image.shape[2] != 4:
        print("Input image does not have an alpha channel")

    # Get the original dimensions
    original_height, original_width = large_image.shape[:2]
    
    scale_factor = 1.0

    # Calculate the scaling factor while maintaining the aspect ratio
    if shape is not None:
        scale_factor = min(shape[0] / original_width, shape[1] / original_height)
    if isinstance(ratio, float) or isinstance(ratio, int):
        scale_factor = ratio
    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image to shape (3000x3000) pixels 
    resized_image = cv2.resize(large_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Save the images as .png
    cv2.imwrite(dst_image_path, resized_image) 
    return (new_width, new_height)

def read_and_resize_large_image_adobe_rgb(src_image_path: str, dst_image_path:str, shape:tuple=(3000, 3000)) -> tuple:
    """
    Reads a large image (TIFF or PNG) and downsizes it to 3000x3000 pixels.

    Parameters:
        image_path (str): Path to the large image file.

    Returns:
        numpy.ndarray: Resized image of size shape 3000x3000 pixels.
    """
    # Load the image
    large_image = Image.open(src_image_path)

    if large_image.mode != 'RGBA':
        large_image = large_image.convert('RGBA')

    # Check if the image has an ICC profile
    icc_profile = large_image.info.get("icc_profile")

    # Define the sRGB IEC61966-2.1 profile
    srgb_profile = ImageCms.createProfile("sRGB")

    # Convert to sRGB if an ICC profile exists and it's not already sRGB
    if icc_profile:
        large_image = ImageCms.profileToProfile(large_image, icc_profile, srgb_profile, outputMode="RGBA")
    else:
        # Assign the sRGB profile if no profile exists
        large_image = ImageCms.profileToProfile(large_image, srgb_profile, srgb_profile, outputMode="RGBA")

    # Resize the image

    # Calculate the scaling factor while maintaining the aspect ratio
    scale_factor = min(shape[0] / large_image.width, shape[1] / large_image.height)

    # Calculate new dimensions
    new_width = int(large_image.width * scale_factor)
    new_height = int(large_image.height * scale_factor)

    resized_img = large_image.resize((1678, 2538), Image.LANCZOS)

    # Save the resized image with the sRGB profile
    serialyzed_profile = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
    resized_img.save(dst_image_path, icc_profile=serialyzed_profile, format='PNG')
    return (new_width, new_height)


def scissors(src_image_path: str, y: int = 0, x: int = 0, h: int = 4000, w: int = 4000):
    img = cv2.imread(src_image_path)

    height, width, channels = img.shape 
    print("Total image: ", height, width, channels)
    rows = int(height / h)
    columns = int(width / w)
    print("Crop by: ", height / h, width / w)

    for x_shift_index in range(4):
        x_shift = int((w / 4) * x_shift_index)
        x = 0 + x_shift 
        for y_shift_index in range(4):
            y_shift = int((h / 4) * y_shift_index)
            y = 0 + y_shift
            for row in range(rows):
                for column in range(columns):
                    crop_h = y + h
                    crop_w = x + w
                    print(y, crop_h, x, crop_w)
                    crop_img = img[y:y+h, x:x+w]
                    image_name=f"upload/full_map_cropped_{row}_{column}_{x_shift_index}_{y_shift_index}.png"
                    print(f"Save at: {image_name}")
                    cv2.imwrite(image_name, crop_img)
                    x += w
                x = 0 + x_shift
                y += h
    
def match_images_and_get_box_sift(big_img, small_img)-> tuple:
    """
    Finds the bounding box of a smaller image within a larger image using the SIFT algorithm.

    Parameters:
        big_img (numpy.ndarray): The larger image.
        small_img (numpy.ndarray): The smaller image to locate.

    Returns:
        tuple: Bounding box coordinates as (x_min, y_min, x_max, y_max), or None if no match is found.
    """
    # Initialize the SIFT detector
    #sigma=1.6, nOctaveLayers=4, nfeatures=1000, sigma=1.6
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors in both images
    kp1, des1 = sift.detectAndCompute(small_img, None)
    kp2, des2 = sift.detectAndCompute(big_img, None)

    # Use FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using KNN
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply the ratio test to keep good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    matched_kps = []
    # Ensure there are enough good matches
    if len(good_matches) > 20:
        # Extract matched keypoints' coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        matched_kps = [kp2[m.trainIdx] for m in good_matches]

        # Compute the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Get the dimensions of the smaller image
            h, w = small_img.shape[:2]

            # Define the corners of the smaller image
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

            # Transform the corners to the larger image's coordinate space
            transformed_corners = cv2.perspectiveTransform(corners, M)

            # Get the bounding box coordinates
            x_min = int(np.min(transformed_corners[:, 0, 0]))
            y_min = int(np.min(transformed_corners[:, 0, 1]))
            x_max = int(np.max(transformed_corners[:, 0, 0]))
            y_max = int(np.max(transformed_corners[:, 0, 1]))

            return ((x_min, y_min, x_max, y_max), matched_kps)

    return None


def match_images_and_get_box_orb(big_img, small_img):
    """
    Finds the location of small_img (fragment) in big_img (scene) using ORB and returns 
    the bounding box coordinates of the detected region.

    Args:
        big_img (numpy.ndarray): Larger scene image (where the fragment will be searched).
        small_img (numpy.ndarray): Smaller fragment image to find in the scene.

    Returns:
        list: List of bounding box coordinates as [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
              or None if no good match is found.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect and compute keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(small_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(big_img, None)

    # Use BFMatcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Check if enough matches are found
    if len(matches) < 4:  # Minimum for homography
        print("Not enough matches found!")
        return None

    # Extract locations of matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("Homography could not be computed!")
        return None

    # Get dimensions of the small image
    h, w = small_img.shape[:2]

    # Define the points of the bounding box in the small image
    points = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    # Transform the points to the big image's coordinate space
    transformed_points = cv2.perspectiveTransform(points, M)

    # Convert to a list of tuples
    box_coordinates = [(int(x), int(y)) for [x, y] in transformed_points.reshape(-1, 2)]

    return box_coordinates

def read_and_prepare_image(img_path: str) -> tuple:
    
    img = cv2.imread(img_path) #, cv2.IMREAD_UNCHANGED
    # Experiment with grayscale
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)

    # Apply Laplacian filter to enhance edges
    laplacian = cv2.Laplacian(enhanced_img, cv2.CV_64F)
    edge_enhanced_img = cv2.convertScaleAbs(laplacian)
    
    # scale_factor = 1.0
    # img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    # gray_img = cv2.resize(gray_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    # edge_enhanced_img = cv2.resize(edge_enhanced_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    return (img, gray_img, edge_enhanced_img)


def open_images_and_orb_find_bbox(big_img_path: str, small_img_path: str) -> None:
    if os.path.exists(big_img_path) == False or os.path.exists(small_img_path) == False:
        return None
            
    big_img, big_gray_img, big_edge_enhanced_img = read_and_prepare_image(big_img_path)    
    small_img, small_gray_img, small_edge_enhanced_img = read_and_prepare_image(small_img_path)
    
    box_coords = match_images_and_get_box_orb(big_gray_img, small_gray_img)
    full_map = cv2.polylines(big_img, [np.array(box_coords, np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)

    dst_img_path = f"upload/result_{str(uuid4())}.png"
    print(f"Found results at: {dst_img_path}")
    cv2.imwrite(dst_img_path, full_map)
    return dst_img_path

def open_images_and_sift_find_bbox(big_img_path: str, small_img_path: str) -> None:
    if os.path.exists(big_img_path) == False or os.path.exists(small_img_path) == False:
        return None
            
    big_img, big_gray_img, big_edge_enhanced_img = read_and_prepare_image(big_img_path)    
    small_img, small_gray_img, small_edge_enhanced_img = read_and_prepare_image(small_img_path)
    
    result = match_images_and_get_box_sift(big_img, small_img)
    if result is None:
        return None
    box, matched_kps = result
    
    # Convert box to coordinates
    x_min, y_min, x_max, y_max = box
    start_point = (x_min, y_min) 
    end_point = (x_max, y_max)
    print(f"found box: {box}")
    full_map = cv2.rectangle(big_img, start_point, end_point, color=(85 , 75, 255), thickness=10)
    full_map = cv2.drawKeypoints(full_map, matched_kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    dst_img_path = f"upload/result_{str(uuid4())}.png"
    print(f"Found results at: {dst_img_path}")
    cv2.imwrite(dst_img_path, full_map)
    return dst_img_path
        

def open_images_and_template_find_bbox(big_img_path: str, small_img_path: str, first_look: bool=False) -> list:
    """
    Finds the location of small_img (fragment) in big_img (scene) using OpenCV matchTemplate and returns 
    list of the bounding boxes coordinates of the detected region, paths, confidence.

    Args:
        big_img (numpy.ndarray): Larger scene image (where the fragment will be searched).
        small_img (numpy.ndarray): Smaller fragment image to find in the scene.
        first_look (Bool): Use only one method

    Returns:
        list: List
    """

    if os.path.exists(big_img_path) == False or os.path.exists(small_img_path) == False:
        return None
            
    big_img, big_gray_img, _ = read_and_prepare_image(big_img_path)    
    _, small_gray_img, _ = read_and_prepare_image(small_img_path)
    
    # For matchTemplate use GRAYSCALE image
    template = small_gray_img
    
    if len(template.shape) == 3:
        h, w, c = template.shape
    else:
        h, w = template.shape

    # Use other methods for debug
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    methods_desc = ["cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR", "cv2.TM_CCORR_NORMED", "cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]
    methods_indx = 0

    if first_look:
        methods = [cv2.TM_CCOEFF]
        methods_desc = ["cv2.TM_CCOEFF"]
    
    results = []
    for method in methods:
        big_img_copy = big_gray_img.copy()
        result = cv2.matchTemplate(big_img_copy, template,  method)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            confidence = 1 - min_val
            location = min_loc
        else:
            normalized_result = (result - min_val) / (max_val - min_val)
            confidence = (max_val - min_val) / max_val            
            location = max_loc

        # Find locations where the confidence exceeds the threshold, for debug purpuses
        threshold = 0.999
        locations = np.where(normalized_result >= threshold)
        # Display all matching points
        for pt in zip(*locations[::-1]):  # Swap x and y for OpenCV compatibility
            print(f"Match found at {pt} with confidence >= {threshold}")
            bottom_right = (pt[0] + w, pt[1] + h)
            cv2.rectangle(big_img, pt, bottom_right, (255, 0, 0), 1)
        
        # Convert coordinates to box points
        start_point = list(location)
        end_point = [location[0] + w, location[1] + h]

        print(f"method: {methods_desc[methods_indx]}, confidence: {confidence: 0.2f} box: {start_point}, {end_point}")

        full_map = cv2.rectangle(big_img.copy(), start_point, end_point, color=(85 , 75, 255), thickness=25)
    
        dst_img_path = f"upload/result_{confidence:0.2f}_{methods_desc[methods_indx]}_{str(uuid4())[:5]}.png"
        
        print(f"Result saved at: {dst_img_path}")
        cv2.imwrite(dst_img_path, full_map)
        box = (start_point, end_point)
        results.append((box, confidence, dst_img_path, methods_desc[methods_indx]))
        methods_indx += 1
    return results



def search_for_tile(big_img_path: str, small_img_path: str) -> tuple:
    
    resized_image_path = "upload/resized_map.png"
    zoomed_img_path = "upload/zoomed_map.png"
    ratio = 0.2
    if os.path.exists(resized_image_path) == False:
        read_and_resize_large_image(big_img_path, resized_image_path, shape=None, ratio=ratio)

    results = open_images_and_template_find_bbox(resized_image_path, small_img_path, first_look=True)
    if isinstance(results, list) == False or len(results) != 1:
        raise RuntimeError

    box, confidence, dst_img_path, method = results[0]
    print(box, confidence, dst_img_path)

    origin_img = cv2.imread(big_img_path)

    factor = 1 / ratio
    box = np.array(box) * factor
    box = box.astype(int)
    zoomed_img = origin_img[box[1][0] : box[1][1], box[0][0] : box[0][1]]

    
    cv2.imwrite(zoomed_img_path, zoomed_img)
    results = open_images_and_template_find_bbox(zoomed_img_path, small_img_path, first_look=False)
    for result in results:
        unzoomed_box, confidence, dst_img_path, method = result
        if method == "cv2.TM_CCORR":
            return (box, confidence, dst_img_path, method)




def search_for_tile_top_left_corner(big_img_path: str, small_img_path: str) -> type:
    result = search_for_tile(big_img_path, small_img_path)
    if isinstance(result, tuple):
        box, confidence, dst_img_path, method = result
        return box[0]











