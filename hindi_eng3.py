import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
import base64  # Import base64 module
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
        text = pytesseract.image_to_string(img, lang=language)
        st.subheader(f"Detected Handwritten Text ({language}):")
        st.text_area(f"Text ({language})", text, height=200)

        # Add a button to download the text as a file
        download_button = st.button("Download Text")
        if download_button:
            download_text_as_file(text)

    except Exception as e:
        st.error(f"Error: {e}")

def download_text_as_file(text):
    # Save the text to a temporary file
    with open("detected_text.txt", "w", encoding="utf-8") as file:
        file.write(text)

    # Provide a link to download the text file
    st.markdown(get_binary_file_downloader_html("detected_text.txt", "Download Text"), unsafe_allow_html=True)
    os.remove("detected_text.txt")  # Remove the temporary file after download

def get_binary_file_downloader_html(file_path, file_label='File'):
    with open(file_path, 'rb') as file:
        file_content = file.read()
    b64 = base64.b64encode(file_content).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{file_path}">{file_label}</a>'

def main():
    st.title("Handwritten Text Detection App")

    # Upload the image file
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "gif"])

    # Language selection dropdown
    language = st.selectbox("Select Language", ["eng", "hin"], index=0)  # Default to English

    if uploaded_image:
        detect_handwritten_text(uploaded_image, language)

if __name__ == "__main__":
    main()
