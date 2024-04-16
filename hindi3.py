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

def detect_handwritten_text(image):
    try:
        # Open the image
        img = Image.open(image)

        # Preprocess the entire image
        img = preprocess_image(img)

        # Display the uploaded image
        st.image(img, caption="Preprocessed Image", use_column_width=True)

        # Use pytesseract to perform OCR with Marathi language model
        text = pytesseract.image_to_string(img, lang='mar')

        return text

    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.title("Handwritten Text Detection App")

    # Upload the image file
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "gif"])

    if uploaded_image:
        text = detect_handwritten_text(uploaded_image)
        if text:
            st.subheader("Detected Handwritten Text:")
            st.text_area("Text", text, height=200)

            # Add a download button to download the extracted text
            download_button = st.button("Download Text")
            if download_button:
                # Save the text to a temporary file
                with open("extracted_text.txt", "w", encoding="utf-8") as file:
                    file.write(text)
                
                # Provide a link to download the text file
                st.markdown(get_binary_file_downloader_html("extracted_text.txt", "Text file"), unsafe_allow_html=True)
                os.remove("extracted_text.txt")  # Remove the temporary file after download

def get_binary_file_downloader_html(file_path, file_label='File'):
    with open(file_path, 'rb') as file:
        file_content = file.read()
    b64 = base64.b64encode(file_content).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{file_path}">{file_label}</a>'

if __name__ == "__main__":
    main()
