import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

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
    # Convert image to grayscale
    img = _img.convert('L')

    # Enhance image quality
    img = enhance_image(img)

    # Apply thresholding to binarize the image
    img = img.point(lambda x: 0 if x < 140 else 255)

    return img

@st.cache_data
def detect_handwritten_text(image):
    try:
        # Open the image
        img = Image.open(image)

        # Preprocess the entire image
        img = preprocess_image(img)

        # Display the uploaded image
        st.image(img, caption="Preprocessed Image", use_column_width=True)

        # Use pytesseract to perform OCR
        text = pytesseract.image_to_string(img)

        # Display the detected text
        st.subheader("Detected Handwritten Text:")
        st.text_area("Text", text, height=200)

    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.title("Handwritten Text Detection App")

    # Upload the image file
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "gif"])

    if uploaded_image:
        detect_handwritten_text(uploaded_image)

if __name__ == "__main__":
    main()
