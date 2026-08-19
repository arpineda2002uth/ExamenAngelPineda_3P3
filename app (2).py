import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Título de la aplicación
st.title("Clasificador de Imágenes - Ángel Pineda")

# Cargar el modelo entrenado en formato SavedModel
# Asegúrate de subir la carpeta "modelo_cifar10" completa a tu repositorio
model = tf.keras.models.load_model("modelo_cifar10.h5")

# Etiquetas del dataset CIFAR-10
labels = ['avión','auto','pájaro','gato','ciervo','perro','rana','caballo','barco','camión']

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((32,32))
    st.image(img, caption="Imagen subida", use_column_width=True)

    # Preprocesar imagen
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    # Predicción
    pred = model.predict(img_array)
    st.write(f"Predicción: {labels[np.argmax(pred)]}")
    st.write(f"Confianza: {np.max(pred):.2f}")
