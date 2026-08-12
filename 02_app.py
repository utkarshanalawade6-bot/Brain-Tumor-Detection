import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("brain_tumor_model.keras")

# Class names
class_names = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

st.title("🧠 Brain Tumor Detection using CNN")
st.write("Upload an MRI Brain Scan Image")

uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction")
    st.success(f"{predicted_class}")

    st.subheader("Confidence")
    st.write(f"{confidence:.2f}%")