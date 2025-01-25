# -*- coding:utf8 -*-
# !/usr/bin/env python3
# Copyright 2025 Volodymyr Pavliukevych
# Author Volodymyr Pavliukevych

"""This is a POC page representation
"""
import os
import streamlit as st 

from PIL import Image
from uuid import uuid4
from src.helper import open_images_and_sift_find_bbox, read_and_resize_large_image, search_for_tile


# import cv2
# import transformers

# from typing_extensions import Final, TypeAlias

# from typing import (
#     TYPE_CHECKING,
#     Any,
#     Callable,
#     Dict,
#     Generic,
#     Optional,
#     Sequence,
#     Tuple,
#     TypeVar,
#     Union,
# )

class POCPage():
    def __init__(self, kernel) -> None:
        self.kernel=kernel
        self.title = "POC page"
        self.description = None
        self.text_example = """
                            Proof of concept
                            """


    def draw(self):
        st.subheader(self.title)
        if self.description is not None:
            st.markdown(self.description)

class EmptyPage(POCPage):
    def __init__(self, kernel) -> None:
        super().__init__(kernel=kernel)
        self.title = "Empty!"

    def draw(self):
        super().draw()

class WelcomePage(POCPage):
    def __init__(self, kernel) -> None:
        super().__init__(kernel=kernel)
        self.title = "Welcome!"

    def draw(self):
        super().draw()
        
        image_column, welcome_text_column = st.columns(2)
        with image_column:
            our_image = Image.open(os.path.join('images/welcome.png'))
            st.image(our_image)
        with welcome_text_column:
            welcome = """
        Our AI research team is at the forefront of developing a groundbreaking solution for the autonomous control of military drones, ushering in a new era of intelligent unmanned systems. 
        
        Focused on advancing the intersection of artificial intelligence and military technology, our researchers are dedicated to crafting an innovative framework that combines machine learning, computer vision, and decision-making algorithms. 
        
        The primary goal is to create a sophisticated AI system capable of autonomously navigating complex environments, identifying targets, and executing missions with unparalleled precision. Collaborating closely with defense experts and incorporating ethical considerations, our research aims to ensure responsible and secure deployment of AI-driven military drone solutions. 
        
        By pushing the boundaries of technological capabilities, our team strives to contribute to the development of cutting-edge tools that enhance military effectiveness while prioritizing safety, accuracy, and adherence to international norms.            
        """
            st.write(welcome)

class TemplatePage(POCPage):
    def __init__(self, kernel) -> None:
        super().__init__(kernel=kernel)
        self.title = "POC"
        self.resized_map_image_path = "upload/resized_map.png"
        self.origin_map_image_path = "upload/origin_map.png"
        self.tile_image_path: str = None

    def draw(self):
        super().draw()
        option = st.selectbox('Select algorithm', ('OpenCV matchTemplate', 'OpenCV SIFT'))
        

        left_column, center_column, right_column = st.columns(3,  border=True)
        with left_column:
            if os.path.exists(self.resized_map_image_path) == False or os.path.exists(self.origin_map_image_path) == False:
                map_uploaded_file = st.file_uploader("Choose a map image (a big one)", type=['png','jpeg','jpg', 'tif'], accept_multiple_files=False)
                if map_uploaded_file is not None:

                    # Write file on fs
                    with open(self.origin_map_image_path, "wb") as file:
                        file.write(map_uploaded_file.getbuffer())            

                    if os.path.exists(self.origin_map_image_path):
                        st.success("Map image saved")
                    else:
                        st.warning("Uploaded file not found")
                    
                    with st.spinner("Resizing image, Please wait."):
                        # 8394x12698 => 6128x9270, 6349, 4232, 3174, 2538, 2116 => 1678 × 2538
                        tuple_image_size = read_and_resize_large_image(self.origin_map_image_path, self.resized_map_image_path, shape=None, ratio=0.2)
                        print(f"Map resized, saved at: {self.resized_map_image_path} with size: {tuple_image_size}")                    
                        if os.path.exists(self.resized_map_image_path) == False:
                            st.error(f"Can't resize image: '{self.origin_map_image_path}'")
            
            if os.path.exists(self.resized_map_image_path):
                map_image = Image.open(self.resized_map_image_path)
                st.image(map_image, caption='Selected map image', width=400)                
                if st.button("Reset map image", type="primary"):
                    os.remove(self.resized_map_image_path)
                    print(f"Remove resized map: {self.resized_map_image_path}")

        
        tile_uploaded_file = st.file_uploader("Choose a tile image (a small image, tile)", type=['png','jpeg','jpg', 'tif'], accept_multiple_files=False)
        if tile_uploaded_file is not None:
            with center_column:        
                tile_file_path = os.path.join("upload", tile_uploaded_file.name)
                # Write file on fs
                with open(tile_file_path, "wb") as file:
                    file.write(tile_uploaded_file.getbuffer())            
                    print(f"Write tile file at path: '{tile_file_path}'")

                if self.tile_image_path is None and os.path.exists(tile_file_path):
                    self.tile_image_path = tile_file_path
                    tile_image = Image.open(self.tile_image_path)
                    st.image(tile_image, caption='Selected map image', width=300)                

                    st.success("Tile image saved")
                elif self.tile_image_path is not None:
                    st.warning("Tile images was set")

            with right_column:
                with st.spinner("Searching for location."):
                    if option == 'OpenCV matchTemplate':
                        result = search_for_tile(self.origin_map_image_path, self.tile_image_path)
                        if isinstance(result, tuple):
                            box, confidence, dst_img_path, method = result
                            st.write(f"Confidence: {confidence:0.4f}, method: {method}")
                            our_image = Image.open(dst_img_path)
                            st.image(our_image, caption='Result image', width=400)

                    elif option == 'OpenCV SIFT':
                        path = open_images_and_sift_find_bbox(self.resized_map_image_path, self.tile_image_path)
                        if isinstance(path, str):
                            our_image = Image.open(path)
                            st.image(our_image, caption='Result image', width=400)
                    else:
                        st.warning("Location not found")
                    
