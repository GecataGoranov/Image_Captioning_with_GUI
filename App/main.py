import streamlit as st
import pandas as pd
import torch
import os

from utils.utils import TextEncoder, retrieve_images


@st.cache_data
def prepare_data():
    embeds = pd.read_pickle("../saved_data/embeddings_dataframe.pkl")
    model = TextEncoder(embed_size=256)
    model.load_state_dict(torch.load("../saved_data/text_encoder_epoch_50.pt", map_location=torch.device("cpu")))
    model.eval()

    return embeds, model


def main():
    st.markdown("""
        <style>
            .stTextInput > label {
                font-size: 1.2rem;
                font-weight: bold;
                color: #4CAF50;
                text-align: center;
            }
            .stTextInput > div > input {
                text-align: center;
                font-size: 1.2rem;
                padding: 0.5rem;
                border-radius: 5px;
                border: 1px solid #4CAF50;
            }
            .stImage {
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 80%;
                height: auto;
            }
            .stImageCaption {
                text-align: center;
                font-size: 0.9rem;
                color: #555;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Image Retrieval Application")
    query = st.text_input("Type your caption:", label_visibility="collapsed")
    if query:
        embeds, model = prepare_data()
        image_names = retrieve_images(model, embeds, query, 15)
        for image_name in image_names:
            image_path = os.path.join("../images", image_name)
            st.image(image_path, caption=image_name, use_container_width=True)


if __name__ == "__main__":
    main()