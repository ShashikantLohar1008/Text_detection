import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np

# Function to detect tables using contours
def detect_tables(img):
    if img is None:
        return []

    # Convert image to grayscale
    if len(img.shape) > 2 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to identify potential tables
    tables = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this threshold based on your image
            tables.append(contour)

    return tables

# Function to detect diagrams using edge density
def detect_diagrams(img):
    if img is None:
        return False

    # Convert image to grayscale
    if len(img.shape) > 2 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Calculate edge density
    height, width = edged.shape
    total_edges = cv2.countNonZero(edged)
    edge_density = total_edges / (height * width)

    # If edge density is above a threshold, consider it as a diagram
    if edge_density > 0.1:  # Adjust this threshold based on your image
        return True
    else:
        return False

@st.cache_data
def enhance_image(img):
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)

    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)

    return img

@st.cache_data
def preprocess_image(_img):
    if _img is None:
        return None

    # Convert image to grayscale
    img = _img.convert('L')

    # Enhance image quality
    img = enhance_image(img)

    # Apply thresholding to binarize the image
    img = img.point(lambda x: 0 if x < 140 else 255)

    return img

def detect_handwritten_text(image):
    try:
        # Open the image
        img = Image.open(image)

        # Preprocess the entire image
        img = preprocess_image(img)

        if img is None:
            return None

        # Display the uploaded image
        st.image(img, caption="Preprocessed Image", use_column_width=True)

        # Use pytesseract to perform OCR
        text = pytesseract.image_to_string(img)

        return text

    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.title("Handwritten Text, Table, and Diagram Detection App")

    # Upload the image file
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "gif"])

    if uploaded_image:
        st.subheader("Original Image:")
        st.image(uploaded_image, caption="Original Image", use_column_width=True)

        text = detect_handwritten_text(uploaded_image)
        if text:
            st.subheader("Detected Handwritten Text:")
            st.text_area("Text", text, height=200)

        # Convert uploaded image to NumPy array
        np_img = np.array(uploaded_image)

        # Detect tables
        try:
            detected_tables = detect_tables(np_img)
            if detected_tables:
                st.subheader("Detected Tables:")
                # Display detected tables (if any)
                for table in detected_tables:
                    st.write("Table Detected")
        except Exception as e:
            st.error(f"Error detecting tables: {e}")

        # Detect diagrams
        try:
            if detect_diagrams(np_img):
                st.subheader("Detected Diagrams:")
                st.write("Diagram Detected")
        except Exception as e:
            st.error(f"Error detecting diagrams: {e}")

if __name__ == "__main__":
    main()
