import pickle

import keras
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Pokemon Type Prediction using DL")

@st.cache_resource
def load_encoder() -> LabelEncoder:
    with open("ll.pkl", 'rb') as fp:
        ll = pickle.load(fp)

    return ll

def to_types(preds):
    return load_encoder().inverse_transform(preds)

@st.cache_resource
def load_model() -> keras.Model:
    model = keras.models.load_model("./pokemon_mnist.keras")
    assert isinstance(model, keras.Model)
    return model


def predict(data):
    model = load_model()
    pred = model.predict(data)
    return pred


st.title("Pokemon Type Detection using Deep Learning Model")
st.markdown("### Pick any pokemon image, we will recognize its type")

pokemon_image = st.file_uploader("Pick any pokemon image")

if pokemon_image:

    st.image(pokemon_image)
    with st.spinner(text="Thinking"):
        img = Image.open(pokemon_image)
        img = img.resize((120, 112))
        img = img.convert("RGBA")
        imgs = np.array([np.array(img) / 255])
        img.close()

        pred = predict(imgs)
        preds = pred.argmax(axis=-1)
        st.success(f"It is of **{to_types(preds)[0]}** Type", icon="✨")
