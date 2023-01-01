import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie
import json

# Creating function to load icon file 
@st.cache()
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    

filepath = "smoking.json"
lottie_json = load_lottiefile(filepath)

# Creating a function to load the dataset
@st.cache() # Decorator from streamlit to keep the returned function in the cache
def get_df():
    df = pd.read_csv("BRFSS.csv", sep=",")
    return df

# Assigning df_smokers for the smokers status topic
df_smokers = get_df()
df_smokers = df_smokers[df_smokers['Topic'] == 'Smoker Status']


st_lottie(lottie_json, height = 100, width = 100)
st.title("Smoking Status")
st.markdown("""---""")

st.markdown("""
One aspect of the BRFSS survey is to collect data on smoking status. 
This page provides an overview of smoking prevalence among adults in the United States over the past decade, broken down by various demographic categories.
The filters provided can be used to further explore smoking prevalence among different population groups, such as by gender, age, race/ethnicity, education level, and income. 
This information can be used to better understand the factors that contribute to smoking behavior.
""")


st.subheader("Data Visualization")
tab1, tab2, tab3 = st.tabs(["Group Category", "Location", "Conclusion"]) # Dividing into 3 tabs

# ----------------------Data viz for group category---------------------- #

with tab1:

    st.write("""
    The relative frequency is represented by the percentage which is based on the sum of each group category of respondents in the year of BRFSS Survey. \n
    e.g., frequency of male smokers in 2011 is male respondents that are smokers among the total male respondents in the \'smokers status\' topic in 2011.
    """)

    col5, col6 = st.columns(2) # make 2 columns for selectbox
    with col5:
        group_cat = st.selectbox(
            "Group category",
            [
                "Age Group",
                "Education Attained",
                "Race/Ethnicity",
                "Gender",
                "Household Income",
                "Overall",
            ],
        )
    with col6:
        year1 = st.selectbox("Year", df_smokers["Year"].unique())

    radio = st.radio('Select the graph', ['Line graph','Bar graph'], horizontal = True) # selection fro type of graph

    # Specifying df_smokers according to the group category
    df_cat = df_smokers[df_smokers.Class_Category == group_cat]
    data_plotly = df_cat[(df_cat.Response == "Yes")]

    
    # Condition if the Line graph is selected produce the line graph, else produce the bar graph
    # Line graph: plotly
    # Bar graph: seaborn
    if radio == 'Line graph':

        fig = px.line(
            data_plotly, x="Year", y="Percentage", color="Category", template="seaborn", width=700, height=400,
            title=f"Frequency distribution of smokers in the USA across the {group_cat}",
            hover_data=['Percentage', 'Sample_Size', 'Total_SS', 'Year'],
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        if group_cat in ['Gender', 'Overall']:
            chart = sns.barplot(x='Category', y='Percentage',
                                data = data_plotly[data_plotly['Year'] == year1], palette='Set2', ci=None, width=0.2)
        else:
            chart = sns.barplot(x ='Category', y='Percentage',
                                data = data_plotly[data_plotly['Year'] == year1], palette='Set2', ci=None, width=0.7)
        plt.title(
            f"Frequency distribution of smokers in the USA across the {group_cat} in {year1}", fontsize=15, pad=20, weight='semibold')

        if group_cat == 'Race/Ethnicity':
            plt.xticks(rotation=90)

        for p in chart.patches:
            chart.annotate("{:,.2f}".format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2, p.get_height()),
                           ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=6)
        st.pyplot(fig)



    # Adding the expander to see the raw number 
    raw_data = st.expander("See the raw number")
    with raw_data:
        tab5, tab6 = st.tabs(['Total Sample Size', 'Sample Size'])
        with tab5:
            st.write(f"The total number of the {group_cat} respondents in the year 2011 - 2020")
            st.dataframe(data_plotly.pivot(index = 'Category', columns= 'Year', values = "Total_SS"))

        with tab6:
            st.write("The number of the respondents who are smokers in the year 2011 - 2020")
            st.dataframe(data_plotly.pivot(index = 'Category', columns= 'Year', values = "Sample_Size"))

   
# ----------------------Data viz for location category---------------------- #

with tab2:

    st.subheader("Smokers Distribution Based on The Location")
    st.write("Frequency distribution of smokers across the USA states ranked by the highest smokers prevalence.")
    st.write("\n\n\n")


    # token = "pk.eyJ1IjoieXVzdWYtZGp1YW5kYSIsImEiOiJjbGM1eWo1djAwdGpwM29sOHRjMzJxNXBjIn0.a74Lmpnj7Tkk41lUhF44fA"
    # fig = px.scatter_mapbox(df_loc, lat="latitude", lon="longitude", 
    # hover_name= "Locationdesc", color="Percentage_Loc", size = 'Total_SS_Loc',
    # color_continuous_scale=px.colors.cyclical.IceFire, size_max=200,opacity = 0.3, zoom=4, width = 700, height = 500,
    # ).update_layout(mapbox_accesstoken=token)

    # st.plotly_chart(fig, use_container_width= False)


    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.selectbox("Select year", df_smokers["Year"].unique())
    with col2:
        states = st.selectbox("Number of states", [3,5,10])
    with col3:
        rank = st.selectbox("Sort the frequency by", ["Lowest", "Highest"] )

    sorting = rank == "Lowest"



    @st.cache()
    def get_loc_df():
        df_loc = pd.read_csv("brfss_smokers.csv", sep = ",")
        return df_loc
    
        
    df_loc = get_loc_df()
    df_loc = df_loc.loc[df_loc['Response'] == 'Yes']
    df_loc = df_loc[df_loc["Year"] == year].sort_values(by = 'Percentage_Loc', ascending = sorting).reset_index().head(states)



    subplots = make_subplots(
        rows= len(df_loc),
        cols=1,
        subplot_titles=[x for x in df_loc['Locationdesc']],
        shared_xaxes=True,
        print_grid=False,
        vertical_spacing=(0.45 / len(df_loc)),
    )

    for i, j in df_loc.iterrows():
        subplots.add_trace(dict(
            type='bar',
            orientation='h',
            y=[j["Locationdesc"]],
            x=[j["Percentage_Loc"]],
            text=[f'{j["Percentage_Loc"]}%'],
            hoverinfo='text',
            textposition='auto',
            marker=dict(
                color = [j["Percentage_Loc"]],
                colorscale = "purples",
            ),
        ), i+1, 1)

    subplots['layout'].update(
        showlegend=False,
    )
    _ = subplots['layout'].update(
        width=550,
    )
    for x in subplots["layout"]['annotations']:
        x['x'] = 0
        x['xanchor'] = 'left'
        x['align'] = 'left'
        x['font'] = dict(
            size=15,
        )
    for axis in subplots['layout']:
        if axis.startswith('yaxis') or axis.startswith('xaxis'):
            subplots['layout'][axis]['visible'] = False

    subplots['layout']['margin'] = {
        'l': 0,
        'r': 0,
        't': 20,
        'b': 1,
    }
    height_calc = 45 * len(df_loc)
    height_calc = max([height_calc, 350])
    subplots['layout']['height'] = height_calc
    subplots['layout']['width'] = height_calc



    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(subplots, use_container_width= True)

    with col4:
        c = st.container()
        map = c.expander("See Map")
        with map:
            st.map(df_loc, zoom = 0.7, use_container_width = True)


  

with tab3:

    st.subheader("Conclusion")


    import scipy.stats as stats

    
    st.markdown("""
    According to the most recent data from the BRFSS, the prevalence of smoking among adults in the United States is approximately **14%** in 2020. 
    This means that about 1 in 8 adults in the country are current smokers. 
    However, as shown in the graph, there is significant variation in smoking rates among different population groups.\n
     
    Chi-square test of independence is used to determine if there is a significant association between smoking status and group category as well as the location.
    All the group category has shown a significant association with the smoker status (p<0.05) which indicates the smoking status rate is influenced by the population groups
    such as age group, education level, race/ethnicity, gender, household income, and location.
    """)
    
    # Loading dataframe for crosstab based on group category
    df_statistics = get_df()
    df_statistics = df_statistics[df_statistics['Topic'] == 'Smoker Status']
    df_statistics = df_statistics.groupby(['Class_Category', 'Category', 'Response'])['Sample_Size'].sum().reset_index()
    
    # Loading dataframe for crosstab based on the location
    df_stat_loc = get_loc_df()
    df_stat_loc = df_stat_loc.groupby(['Locationdesc', 'Response'])['Sample_Size'].sum().reset_index()

    statistics = st.expander("See chi-square results")

    with statistics:
        st.write("Cumulative sum of the sample size from 2011 - 2020 is used to calculate the chi-square test")

        group_cat = st.selectbox("Select group category", [
                                 'Age Group', 'Education Attained', 'Gender', 'Household Income', 'Race/Ethnicity', 'Location'])

        
        if group_cat == 'Location':

            crosstab = pd.crosstab(index = df_stat_loc.Locationdesc, columns = df_stat_loc.Response,
                                   values = df_stat_loc.Sample_Size, aggfunc = sum, margins = True)

            # Perform chi-square test

            stat, p, dof, expected = stats.chi2_contingency(crosstab)

            st.write("Chi-square statistic: {:.3f}".format(stat))
            st.write("p-value: {:.3f}".format(p))
            st.write("Degrees of freedom: {:.0f}".format(dof))

        else:

            df_statistics = df_statistics[(df_statistics['Class_Category'] == group_cat)]
            crosstab = pd.crosstab(index=df_statistics.Category, columns=df_statistics.Response,
                                   values=df_statistics.Sample_Size, aggfunc=sum, margins=True)

            # Perform chi-square test
            stat, p, dof, expected = stats.chi2_contingency(crosstab)

            st.write("Chi-square statistic: {:.3f}".format(stat))
            st.write("p-value: {:.3f}".format(p))
            st.write("Degrees of freedom: {:.0f}".format(dof))
            # st.table(pd.DataFrame(expected))

        

        


        
