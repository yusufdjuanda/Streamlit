# Import the dependencies
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# from plotly.subplots import make_subplots
from streamlit_lottie import st_lottie
import json


@st.cache()
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    

filepath = "weight form.json"
lottie_json = load_lottiefile(filepath)
st_lottie(lottie_json, height = 100, width = 100)



# Creating a function to load the dataset
@st.cache() # Decorator from streamlit to keep the returned function in the cache
def get_df():
    df = pd.read_csv("BRFSS.csv", sep=",")
    return df


df = get_df() #loading the dataset



st.title("Topics Covered")
st.markdown("""---""")

st.markdown(f"""
Based on the given dataset, there are **{len(df['Topic'].unique())}** different topics inquired during the BRFSS survey. 
The number of questions may vary in accordance with the topic. The type of questions are open-ended which lead to simple categorical responses.
\n
This page provides an overview of the response proportion from every question with for the corresponding year of the survey conducted.
""")

# ----------------------Filter selection for topic and year---------------------- #

col1, col2 = st.columns(2)
with col1:
    chosen_topic = st.selectbox(
        "Topics",
        df["Topic"].sort_values().value_counts().sort_index().index.tolist(),
    )
with col2:
    year = st.selectbox('Select year', df["Year"].loc[df["Topic"] == chosen_topic].unique())


# ----------------------Questions section---------------------- #

st.subheader("**Question(s)**")
selected_question = df["Question"].loc[(df['Topic'] == chosen_topic) & (df['Year'] == year)]

grammar = "is only" if len(selected_question.unique()) == 1 else "are" # function to return "is" if the number of question equals to one or else return "are" 
st.write(f"There {grammar} {len(selected_question.unique())} question(s) provided. ")

# Listing all the question(s) in the table view
st.table(
    pd.DataFrame(
        selected_question.value_counts().index,
        columns=["Question(s)"],
    )
)

# ----------------------Responses section---------------------- #
st.markdown("""---""")

st.subheader("**Responses**")

question = st.selectbox('Select Question(s)', df["Question"].loc[df["Topic"] == chosen_topic].unique())
# Load the dataframe to match with specific question and year
df_response = df.loc[(df['Class_Category'] == "Overall") & (df['Question'] == question)  & (df['Year'] == year)].groupby('Response')['Percentage'].sum().reset_index()
df_response['mid'] = 'Response' # adding one column containing string 'Response' to appear in the middle of pie chart
fig = px.sunburst(
            df_response,
            path=["mid", "Response", "Percentage"],
            values="Percentage",
            color_discrete_sequence=px.colors.qualitative.Set2,
            maxdepth = 2
        ).update_traces(insidetextorientation='auto')

st.plotly_chart(fig, use_container_width= True)


# ----------------------Group category expander---------------------- #


with st.expander("See group category"):

    sunburst_df = (
    df.groupby(["Class_Category", "Category"])["Total_SS"].sum().reset_index()
    )
    sunburst_df = sunburst_df.replace(["Education Attained", "Household Income"], ["Education", "Income"])
    sunburst_df["mid"] = "Category"

    # to make 2 containers
    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader("Group Category")
        st.write(
            "There are 6 unique group categories representing the additional information of the respondents."
        )

    with col2:
        fig = px.sunburst(
            sunburst_df.loc[sunburst_df["Total_SS"] > 10000000],
            path=["mid", "Class_Category", "Category"],
            values="Total_SS",
            color_discrete_sequence=px.colors.qualitative.Set2,
            maxdepth = 2
        ).update_traces(insidetextorientation='auto')
        
        st.plotly_chart(fig, use_container_width=True)