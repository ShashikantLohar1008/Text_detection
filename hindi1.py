import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# Function to enhance image quality
def enhance_image(img):
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)

    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)

    return img

# Function to preprocess the image
def preprocess_image(_img):
    # Convert image to grayscale
    img = _img.convert('L')

    # Enhance image quality
    img = enhance_image(img)

    # Apply thresholding to binarize the image
    img = img.point(lambda x: 0 if x < 140 else 255)

    return img

# Function to detect handwritten text
def detect_handwritten_text(image):
    try:
        # Open the image
        img = Image.open(image)

        # Preprocess the image
        img = preprocess_image(img)

        # Display the preprocessed image
        st.image(img, caption="Preprocessed Image", use_column_width=True)

        # Use pytesseract to perform OCR
        text = pytesseract.image_to_string(img, lang='hin')

        return text

    except Exception as e:
        st.error(f"Error: {e}")

# Main function
def main():
    st.title("Handwritten Text Detection App")

    # Upload the image file
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "gif"])

    if uploaded_image:
        # Display the preprocessed image
        img = Image.open(uploaded_image)
        img = preprocess_image(img)
        st.image(img, caption="Preprocessed Image", use_column_width=True)

        # Detect handwritten text
        text = detect_handwritten_text(uploaded_image)
        if text:
            st.subheader("Detected Handwritten Hindi Text:")
            st.text_area("Text", text, height=200)

if __name__ == "__main__":
    main()
