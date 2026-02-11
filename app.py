import streamlit as st
from PIL import Image

from classifier import classify_image

st.title("Classificador de Imagens 🐶📸")
st.write("Faça upload de uma imagem e veja o que a IA acha que é.")

uploaded_file = st.file_uploader(
    "Escolha uma imagem",
    type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(uploaded_file, caption="Imagem enviada", use_container_width=True)

if st.button("Classificar"):
    with st.spinner("Analisando imagem..."):
        label, confidence = classify_image(image)

    st.success(f"🧠 Predição: **{label}**")
    st.write(f"Confiança: **{confidence:.2%}**")