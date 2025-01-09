import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import cnn
import time
import requests
import google.generativeai as genai
gemini_api_key = "AIzaSyAKbrUz5go3f6jgBLxvas7xseErc-VtnW4"
# Load the plant disease prediction model
model=cnn.CNN(39)
model_path = "plant_disease_model_1_latest.pt"
model.eval()

class_to_disease = {
                        0: 'Apple___Apple_scab',
                        1: 'Apple___Black_rot',
                        2: 'Apple___Cedar_apple_rust',
                        3: 'Apple___healthy',
                        4: 'Background_without_leaves',
                        5: 'Blueberry___healthy',
                        6: 'Cherry___Powdery_mildew',
                        7: 'Cherry___healthy',
                        8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                        9: 'Corn___Common_rust',
                        10: 'Corn___Northern_Leaf_Blight',
                        11: 'Corn___healthy',
                        12: 'Grape___Black_rot',
                        13: 'Grape___Esca_(Black_Measles)',
                        14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        15: 'Grape___healthy',
                        16: 'Orange___Haunglongbing_(Citrus_greening)',
                        17: 'Peach___Bacterial_spot',
                        18: 'Peach___healthy',
                        19: 'Pepper,_bell___Bacterial_spot',
                        20: 'Pepper,_bell___healthy',
                        21: 'Potato___Early_blight',
                        22: 'Potato___Late_blight',
                        23: 'Potato___healthy',
                        24: 'Raspberry___healthy',
                        25: 'Soybean___healthy',
                        26: 'Squash___Powdery_mildew',
                        27: 'Strawberry___Leaf_scorch',
                        28: 'Strawberry___healthy',
                        29: 'Tomato___Bacterial_spot',
                        30: 'Tomato___Early_blight',
                        31: 'Tomato___Late_blight',
                        32: 'Tomato___Leaf_Mold',
                        33: 'Tomato___Septoria_leaf_spot',
                        34: 'Tomato___Spider_mites Two-spotted_spider_mite',
                        35: 'Tomato___Target_Spot',
                        36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        37: 'Tomato___Tomato_mosaic_virus',
                        38: 'Tomato___healthy'
                    }

api_key=gemini_api_key

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("API key not found in environment variables.")
    
def get_gemini_response(question):
    try:
        model=genai.GenerativeModel('gemini-1.5-flash')
        response=model.generate_content(question)
        return response.text
    except Exception as e:
        st.error(f"Error in generating content: {e}")
        return None

def predict_disease(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Resize the image
    image = image.resize((224, 224))
    
    # Convert the image to a tensor
    input_data = TF.to_tensor(image)
    
    # Adjust tensor dimensions
    input_data = input_data.view((-1, 3, 224, 224))
    
    # Make prediction
    with torch.no_grad():
        output = model(input_data)
    
    # Convert tensor to numpy array
    output = output.detach().numpy()
    
    # Get index of the max value
    index = np.argmax(output)
    
    return class_to_disease[index]

def main():
    st.title("FarmSmartAI")

    st.sidebar.header("Choose a Feature")
    choice = st.sidebar.radio("", ("Conversational Agent", "Plant Disease Prediction"))

    if choice == "Conversational Agent":
        st.header("AI Assistant")
        user_input = st.text_area("Ask a question on Agriculture :")
        if st.button("Ask"):
            response = get_gemini_response(user_input)
            if response:
                st.subheader("Generated Response:")
                st.write(response)
            else:
                st.write("No response received from Gemini API.")

    elif choice == "Plant Disease Prediction":
        st.header("Plant Disease Classifier")
        uploaded_file = st.file_uploader("Upload an image of a plant", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            #st.write("Classifying...")
            prediction = predict_disease(uploaded_file)
            st.write(f"Predicted Disease: {prediction}")

if __name__ == "__main__":
    main()
