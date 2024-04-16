import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
# import base64  # Import base64 module
import os

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

def detect_handwritten_text(image, language):
    try:
        # Open the image
        img = Image.open(image)

        # Preprocess the entire image
        img = preprocess_image(img)

        # Display the uploaded image
        st.image(img, caption="Preprocessed Image", use_column_width=True)

        # Use pytesseract to perform OCR with the selected language model
        if language=='eng':
            text = pytesseract.image_to_string(img, lang=language)
        
        if language=='hin':
            text=pytesseract.image_to_string(img,lang=language)

        # st.subheader(f"Detected Handwritten Text ({language}):")
        # st.text_area(f"Text ({language})", text, height=200)
        return text


    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.title("Handwritten Text Detection App")

    # Upload the image file
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "gif"])

    # Language selection dropdown

    language = st.selectbox("Select Language", ["eng", "hin"], index=0)  # Default to English

    if uploaded_image:
        text=detect_handwritten_text(uploaded_image, language)
        if text:
            st.subheader("detected text:")
            st.text_area("Text",text,height=200)

if __name__ == "__main__":
    main()
