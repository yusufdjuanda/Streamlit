import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots



# Creating a function to load the original dataset
@st.cache()
def get_display_df():
    display_df = pd.read_csv(
        "Behavioral_Risk_Factor_Surveillance_System__BRFSS__Prevalence_Data__2011_to_present_.csv",
        sep=";",
        nrows=20,
    )
    return display_df

# Creating a function to load the transformed dataset
@st.cache() # Decorator from streamlit to keep the returned function in the cache
def get_df():
    df = pd.read_csv("BRFSS.csv", sep=",")
    return df

# Creating a function to load the location dataset
@st.cache()
def get_loc_df():
    df_loc = pd.read_csv("brfss_smokers.csv", sep = ",")
    return df_loc

display_df = get_display_df()

heading = st.container()
heading.image("BRFSS.png", width=250)
heading.title("The Behavioral Risk Factor Surveillance System (BRFSS)")
st.markdown("""---""")
heading.markdown(
    """
**BRFSS** is the nation’s premier system of health-related telephone surveys that collect state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services.
For more information, visit https://www.cdc.gov/brfss.
"""
)

tab1, tab2 = st.tabs(["Dataset", "Data Transformation"])
with tab1:
    st.subheader("Dataset")
    st.markdown(
        "The dataset was provided to us via **Google Drive** and for learning purpose, the original dataset was utilized for further analysis."
    )
    st.markdown('\n')

    display = st.container()
    display.write("**Glance of the first 20 rows of the dataset**")
    display.dataframe(display_df.head(20))
    display.markdown(
        """
    _There are **2.048.467** observations and **27** variables_
    """
    )
    with st.expander("See important variables"):
        st.caption(
            """
            1. **Year**: The year of the survey was conducted
            2. **Location**: States in the USA
            3. **Class**: Class of the topic
            3. **Topic**: Topics of the survey which lead specific questions and responses 
            4. **Question**: The question used for the survey
            5. **Response**: The response from the respondents
            6. **Break_Out**:  Group category of the respondents, which later will be renamed to "Category"
            7. **Break_Out_Category**: Class of the group category, which later will be renamed to "Class_Category"
            8. **Sample_Size**: Number of respondents
        """
        )

with tab2:
    st.subheader("Transformed Dataset")
    st.markdown("""
    From the original dataset, (Exploratory Data Analysis) EDA was performed to produce 2 transformed datasets with some additional variables.
    - General dataset
    - Location dataset for smoking status
    """)
    radio = st.radio('Select the transformed dataset', ['General dataset','Location dataset'], horizontal = True)
    if radio == "General dataset":
        transformed_df = get_df()
        st.dataframe(transformed_df.head(20))
        st.write("_There are **3.8142** observations and **10** variables_")
        st.markdown("""
        **Additional variables**
        - Percentage : Relative frequency of the sample size from each response per number of respondents in every question or topic in the specific year e.g., frequency of male smokers in 2011 is male respondents that are smokers among the total male respondents in the \'smokers status\' topic in 2011
        Percentage = sample size / total sample size 
        - Total_SS : Total respondents of specific question in specific year

        """)
       
    else:
        loc_df = get_loc_df()
        st.dataframe(loc_df.head(20))
        st.write("_There are **1.060** observations and **8** variables_")
        st.markdown("""
        **Additional variables**
        - Percentage_Loc : Relative frequency of the sample size from smoking status response per number of respondents in every location or topic in the specific year 
        Percentage = sample size / total sample size 
        - Total_SS_Loc : Total respondents of specific location in specific year
        - latitude / longitude = Coordinates obtained from 'Geolocation' column

        """)




