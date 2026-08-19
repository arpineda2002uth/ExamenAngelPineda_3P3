import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Clasificador de Imágenes - Ángel Pineda")

# Cargar modelo en formato SavedModel
# Asegúrate de subir la carpeta "modelo_cifar10" completa a tu repo
model = tf.keras.models.load_model("modelo_cifar10")

labels = ['avión','auto','pájaro','gato','ciervo','perro','rana','caballo','barco','camión']

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((32,32))
    st.image(img, caption="Imagen subida", use_column_width=True)

    img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    pred = model.predict(img_array)
    st.write(f"Predicción: {labels[np.argmax(pred)]}")
    st.write(f"Confianza: {np.max(pred):.2f}")
