import streamlit as st
from together import Together
import base64
import os
import imghdr

# Class to process images
class ImageProcessor:
    def __init__(self, api_key):
        self.client = Together(api_key=api_key)
        self.prompt = """Convert the provided image into Markdown format.\nEnsure that all page content is included, such as headers, footers, subtexts, images (with alt text if possible), tables, and any other elements.\n\nRequirements:\n\n- Markdown only output: return only the Markdown content without any additional explanations or comments.\n- No Delimiters: Do not use code boundaries or delimiters like ```markdown.\n- Complete Content: Do not omit any part of the page, including headers, footers, and subtext.\n"""
        self.model = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"

    def get_mime_type(self, image_path):
        """Determine MIME type based on the actual image format"""
        img_type = imghdr.what(image_path)
        if img_type:
            return f'image/{img_type}'
        # Fallback for detection based on file extension
        extension = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')

    def encode_image(self, image_path):
        """Encode the image in base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path):
        """Analyze the image using the Together API"""
        base64_image = self.encode_image(image_path)
        mime_type = self.get_mime_type(image_path)

        # Request to the API
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            stream=True,
        )

        # Process the streaming response
        response_text = ""
        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, 'content') else None
                if content:
                    response_text += content
        return response_text

# Streamlit Interface
st.title("OCR with Together AI")

# Input for API key
api_key = st.text_input("Enter your Together API key:", type="password")

# Image upload
uploaded_file = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg", "gif", "webp"])

if api_key and uploaded_file:
    with st.spinner("Processing the image..."):
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the image
        processor = ImageProcessor(api_key)
        try:
            result = processor.analyze_image(temp_file_path)
            st.success("Analysis completed!")
            st.text_area("OCR Result in Markdown:", result, height=300)
        except Exception as e:
            st.error(f"Error processing the image: {e}")
        finally:
            os.remove(temp_file_path)  # Remove the temporary file
