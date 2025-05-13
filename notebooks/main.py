# -*- coding:utf8 -*-
# !/usr/bin/env python3
# Copyright 2025 Volodymyr Pavliukevych
# Author Volodymyr Pavliukevych

"""This is a POC representation
"""
import os 
import cv2
import zipfile
import requests
from tqdm import tqdm
from uuid import uuid4
from itertools import combinations
from math import comb
import shutil

class ImageProcessor:
	"""
	A class to process images from a dataset, including downloading, unzipping, and feature extraction using SIFT.
	"""
	def __init__(self, dataset_folder_name:str='dataset', dataset_url:str=None, images_for_process: list = ['.tif']):
		self.dataset_folder_name = dataset_folder_name
		self.improved_dataset_folder_name = "improved_dataset"
		self.dataset_url = dataset_url
		self.images_for_process = images_for_process
		self.images_descriptions = []

	@property
	def dataset_path(self):
		return os.path.join(os.path.dirname(__file__), self.dataset_folder_name)

	@property
	def improved_dataset_path(self):
		return os.path.join(os.path.dirname(__file__), self.improved_dataset_folder_name)
	
	def read_and_prepare_image(self, img_name: str) -> tuple:
		"""
		Read and prepare an image for processing.
		"""
		img_path = os.path.join(self.dataset_path, img_name)
		img = cv2.imread(img_path) #, cv2.IMREAD_UNCHANGED

		# Get the original dimensions
		original_height, original_width = img.shape[:2]
				
		scale_factor = 0.50
		ratio = None
		shape = None
		# Calculate the scaling factor while maintaining the aspect ratio
		if shape is not None:
			scale_factor = min(img.shape[0] / original_width, img.shape[1] / original_height)
		if isinstance(ratio, float) or isinstance(ratio, int):
			scale_factor = ratio
		# Calculate new dimensions
		new_width = int(original_width * scale_factor)
		new_height = int(original_height * scale_factor)
		img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

		# Experiment with grayscale
		gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
		gray_img = cv2.resize(gray_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
				

		# Enhance contrast using CLAHE
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		enhanced_img = clahe.apply(gray_img)

		# Apply Laplacian filter to enhance edges
		laplacian = cv2.Laplacian(enhanced_img, cv2.CV_64F)
		edge_enhanced_img = cv2.convertScaleAbs(laplacian)
		edge_enhanced_img = cv2.resize(edge_enhanced_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
	
		return (img, gray_img, edge_enhanced_img)
	
	def load_dataset(self) -> str:
		"""
		Load the dataset from the specified URL if it doesn't exist locally.
		"""
		if os.path.exists(self.dataset_path):
			return self.dataset_path
		# Download the dataset if it doesn't exist
		if not os.path.exists(os.path.join(self.dataset_path, 'dataset.zip')):
			self.download_dataset()

		# Unzip the dataset
		self.unzip_dataset()
		return self.dataset_path

	def download_dataset(self):
		"""
		Download the dataset from the specified URL.
		"""
		response = requests.get(self.dataset_url, stream=True)
		total_size = int(response.headers.get('content-length', 0))
		block_size = 1024
		with open(os.path.join(os.path.dirname(__file__), 'dataset.zip'), 'wb') as file:
			for data in tqdm(response.iter_content(block_size), total=total_size, unit='KB', unit_scale=True):
				file.write(data)
	
	def unzip_dataset(self):
		with zipfile.ZipFile(os.path.join(self.dataset_path, 'dataset.zip'), 'r') as zip_ref:
			zip_ref.extractall(self.dataset_path)

	def process_images(self):
		"""
		Process images in the dataset to extract SIFT features.
		"""
		sift = cv2.SIFT_create(contrastThreshold=0.01)

		# Process the images
		for _, _, files in os.walk(self.dataset_path):
			for file_path in tqdm(files, desc="Processing images", unit="file"):
				file_type = os.path.splitext(file_path)[1]
				if file_type not in self.images_for_process:
					continue

				_, img, _ = self.read_and_prepare_image(file_path)
				if img is None:
					print(f"Failed to load image at path: {file_path}")
					continue

				keypoints, descriptors = sift.detectAndCompute(img, None)
				if descriptors is None:
					print(f"Failed to compute descriptors for image at path: {file_path}")
					continue

				self.images_descriptions.append({
					# 'image': img,
					'path': os.path.join(self.dataset_path, file_path),
					'image_name': file_path,
					'keypoints': keypoints,
					'descriptors': descriptors
				})

	def compare_sift_features(self, ratio_thresh=0.75, match_thresh=100):
		"""
		Compare SIFT features of images in the dataset and find similar pairs.
		"""
		bf = cv2.BFMatcher()
		similar_pairs = []
		zones = []

		total_pairs = comb(len(self.images_descriptions), 2)
		for img1, img2 in tqdm(combinations(self.images_descriptions, 2), desc="Comparing images", unit="pair", total=total_pairs):
			# Compare the descriptors of the two images
			des1 = img1['descriptors']
			des2 = img2['descriptors']

			kps1 = img1['keypoints']
			kps2 = img2['keypoints']

			matches = bf.knnMatch(des1, des2, k=2)

			# Lowe's ratio test
			good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
			if False:
				matched_kps = [kps2[m.trainIdx] for m in good_matches]
				
				image_1 = cv2.drawKeypoints(img1['image'], matched_kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
				image_2 = cv2.drawKeypoints(img2['image'], matched_kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
				
				cv2.imwrite(f"{img1['path']}_keypoints.png", image_1)
				cv2.imwrite(f"{img2['path']}_keypoints.png", image_2)

			if len(good_matches) >= match_thresh:
				found_corelated_zone = False
				for zone in zones:
					extended_zone = zone.copy()
					if img1['image_name'] in zone or img2['image_name'] in zone:
						found_corelated_zone = True
						if img1['image_name'] not in zone:
							extended_zone.add(img1['image_name'])
						if img2['image_name'] not in zone:
							extended_zone.add(img2['image_name'])
						zones.remove(zone)
						zones.append(extended_zone)
						break
				if found_corelated_zone == False:
					zone = set()
					zone.add(img1['image_name'])
					zone.add(img2['image_name'])
					zones.append(zone)
				
				similar_pairs.append((img1['image_name'], img2['image_name'], len(good_matches)))
		
		# Save non paired images
		for image in tqdm(self.images_descriptions):
			image_name = image['image_name']
			found_corelated_zone = False
			for zone in zones:
				if image_name in zone:
					found_corelated_zone = True
					break
					
			if found_corelated_zone == False:
				zone = set()
				zone.add(image_name)
				zones.append(zone)

		# merge similar zones
		for i in range(len(zones)):
			for j in range(i + 1, len(zones)):
				if zones[i].intersection(zones[j]):
					print(f"Zones {i} and {j} are similar: {zones[i]} and {zones[j]}")
					zones[i] = zones[i].union(zones[j])
					zones.pop(j)
					break

		
		if os.path.exists(self.improved_dataset_path) == False:
			os.mkdir(self.improved_dataset_path)
			
		for zone_index, zone in enumerate(zones):
			print(f"Zone {zone_index}: {zone}")
			for photo_index, image_name in enumerate(zone):
				src_image_path = os.path.join(self.dataset_path, image_name)
				dst_image_path = os.path.join(self.improved_dataset_path, f"{zone_index}_{photo_index}_{image_name}.tif", )
				shutil.copy(src_image_path, dst_image_path)

		return similar_pairs


def main():

	dataset_url = "https://orients-ai-artefacts.s3.eu-north-1.amazonaws.com/selection-tasks/landc02/dataset.zip"
	image_processor = ImageProcessor(dataset_url=dataset_url)
	# Load the dataset
	_ = image_processor.load_dataset()
	image_processor.process_images()
	similar_pairs = image_processor.compare_sift_features(ratio_thresh=0.75)
	print("Similar pairs of images:")
	for img1, img2, num_matches in similar_pairs:
		print(f"Image 1: {img1}, Image 2: {img2}, Matches: {num_matches}")

if __name__ == '__main__':
	main()