import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import json
from streamlit_lottie import st_lottie

st.set_page_config(layout="wide")

@st.cache()
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

filepath = "employee-getting-customer-requirements.json"
lottie_json = load_lottiefile(filepath)


st_lottie(lottie_json, height = 300, width = 300)

st.title("About the Project")

st.markdown("""
The project is part of the final presentation of _Basic Programming Class_ Fall 2022 from **Taipei Medical University.**
""")
st.markdown("""---""")
st.subheader("Dataset")
st.markdown("""
The Behavioral Risk Factor Surveillance System (BRFSS) dataset was assigned to us to perform further analysis.
""")

st.subheader("Objective")
st.markdown("""
1. To perform Exploratory Data Analysis (EDA) on the BRFSS dataset and to have an overview of the content.
2. To explore further about the smoking status frequency distribution based on different group population.
""")
st.markdown("""---""")
st.subheader("Group 3")
st.markdown("""
- Yusuf Maulana - M610111010
- Lutvi Vitria Kadarwati - M850111006
- Harold Arnold Chinyama  - M610111011
""")
for i in range(5):
    st.write("")
st.markdown("""---""")
st.write("Should you have any feedback or questions, please contact m610111010@tmu.edu.tw")