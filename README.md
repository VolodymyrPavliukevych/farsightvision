### README: POC for Image Matching Using OpenCV

---

#### **Project Overview**
This project is a Proof of Concept (POC) for using OpenCV to match a smaller, potentially rotated image within a larger one. It leverages two powerful OpenCV features:
1. **`matchTemplate`**: A method for basic template matching.
2. **SIFT (Scale-Invariant Feature Transform)**: A feature-based approach to detect and match keypoints, enabling robust handling of rotations and scale differences.

---

#### **Features**
- Template matching using OpenCV's `cv2.matchTemplate`.
- Feature-based image matching with SIFT for detecting smaller, rotated images in larger maps.
- Handling of high-resolution images with optimization techniques.
- Preprocessing options for grayscale conversion and noise reduction.

---

#### **Prerequisites**
- Python 3.12
- OpenCV 4.x or later
- NumPy

Install dependencies:
```bash
pip install opencv-python opencv-contrib-python numpy
```

---

#### **Usage**
1. Clone this repository:
   ```bash
   git clone https://github.com/VolodymyrPavliukevych/farsightvision
   cd farsightvision
   ```

2. Run the script:
   ```bash
   python streamlit run app.py
   ```

3. Input the paths for:
   - The larger image (e.g., map or background).
   - The smaller image to be found.

4. Choose the desired method:
   - **Template Matching**: Quick but sensitive to scale and rotation.
   - **SIFT Matching**: Robust to scale, rotation, and other transformations.

---

#### **Key Algorithms**

1. **Template Matching**:
   - Efficient for finding exact matches.
   - Sensitive to scale and rotation changes.
   - Example method: `cv2.matchTemplate` with `cv2.cv2.TM_CCORR`.

2. **SIFT**:
   - Detects and matches features using keypoints.
   - Handles rotation, scaling, and partial occlusions.
   - Requires OpenCV with the `contrib` module enabled.

---

#### **Directory Structure**
```
├── app.py                 # Main script to run the POC
├── helper.py              # Utility functions for preprocessing and optimization
├── upload/                # Directory to store input images
│   ├── large_image.jpg
│   ├── small_image.jpg
│   └── ...
├── output/               # Output results (cropped matches, keypoints visualization, etc.)
├── requirements.txt      # Dependencies for the project
└── README.md             # Project documentation
```

---

#### **Results**
- The output includes:
  - The location of the smaller image in the larger one.
  - Visualization of matched keypoints (SIFT).
  - Metrics like confidence scores for `matchTemplate`.

---

#### **To Do**
1. Add support for **magnetic and GPS orientation** to improve location matching.
2. Implement preprocessing to **remove unclear, empty, or irrelevant tiles** from the dataset, reducing computation time.
3. Cache the **features of the larger image** to avoid redundant computations across multiple runs.
4. Experiment with **different grayscale conversion methods** for better results in varying lighting conditions.
5. Complete the **ORB-based POC version** as an alternative feature-matching algorithm.
6. Integrate support for **rotating images** to handle cases where the smaller image is rotated in the larger one.
7. Improve the project’s **metrics** to make them clearer and more comprehensive.

---

#### **Contribution**
Feel free to contribute by:
- Adding new feature-matching algorithms.
- Optimizing existing methods for speed and accuracy.
- Expanding preprocessing options.

---

#### **License**
This project is licensed under the MIT License.

Let me know if you need any additions or edits to this README!