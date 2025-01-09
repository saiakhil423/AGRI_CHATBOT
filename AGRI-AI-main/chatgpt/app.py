import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import torch

# Load Hugging Face model for disease identification
model_name = "KissanAI/Dhenu-vision-lora-0.1"
disease_identifier = pipeline("image-classification", model=model_name)

# Streamlit UI
st.title("Crop Disease Detection App")
st.write("Upload an image of the crop to identify disease, symptoms, cure, prevention, and severity.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process and display image if uploaded
if uploaded_image is not None:
    # Open image and display in Streamlit
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for the model (resize, convert to RGB, normalize)
    image = image.resize((224, 224)).convert("RGB")  # Resize to standard input size
    image_np = np.array(image) / 255.0  # Normalize to [0, 1]
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor
    
    # Make prediction
    st.write("Analyzing image for diseases...")
    results = disease_identifier(image_tensor)

    # Parse and display results
    if results:
        disease_info = results[0]
        disease_name = disease_info['label']
        confidence = disease_info['score']
        
        # Sample details (replace with actual model outputs if available)
        st.write(f"**Disease Identified**: {disease_name} (Confidence: {confidence:.2f})")
        st.write(f"**Symptoms**: Placeholder for symptoms related to {disease_name}.")
        st.write(f"**Cure**: Suggested treatments for {disease_name}.")
        st.write(f"**Prevention Methods**: Ways to prevent {disease_name} in future.")
        st.write(f"**Severity**: Based on model or expert inputs for {disease_name}.")

    else:
        st.write("No disease detected, or unable to classify the image.")
