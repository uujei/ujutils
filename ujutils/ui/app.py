import pandas as pd
import numpy as np
import requests
import streamlit as st
from PIL import Image
from ujutils.misc import _estimate_root

st.set_page_config(
    layout="wide", page_title="Page Title",
)

df = pd.read_csv('labels.csv')
root = _estimate_root(df['filepath'])

with st.sidebar:
    st.header("Configuration")
    with st.form(key="grid_reset"):
        n_photos = st.slider("Number of cat photos:", 4, 20, 12)
        n_cols = st.number_input("Number of columns", 2, 8, 4)
        st.form_submit_button(label="Reset images and layout")
    with st.expander("About this app"):
        st.markdown(f"[Preview] {root}")
    st.caption("Source: https://cataas.com/#/")

st.title(f"[Preview] {root}")
st.title("Title")
st.header("Header")
st.subheader("Subheader")
st.caption(
    ' '.join([
        "You can display the image in full size by hovering it",
        "and clicking the double arrow"
    ])
)

images = []
for idx in df.index[:n_photos]:
    try:
        images.append(
            {
                'image': Image.open(df.at[idx, 'filepath']),
                'caption': f"Image No. {idx+1}"
            }
        )
    except Exception as ex:
        images.append(
            {
                'image': None,
                'caption': f"Broken {ex}"
            }
        )

n_rows = 1 + len(images) // n_cols
rows = [st.container() for _ in range(int(n_rows))]

cols_per_row = [r.columns(n_cols) for r in rows]

for i, img in enumerate(images):
    with rows[i // n_cols]:
        if img['image'] is not None:
            cols_per_row[i // n_cols][i % n_cols].image(**img)
