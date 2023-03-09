import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import shap
import joblib


### Config
st.set_page_config(
    page_title="Project",
    page_icon=" ",
    layout="wide"
)


### App
st.title("Dashboarding for the scoring project 🎨")

st.markdown(""" Welcome to the dashboard for the scoring project 👇
""")

st.markdown("---")

DATA_URL = "data.csv"
@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data.drop('Unnamed: 0',axis=1,inplace=True)
    data.set_index('SK_ID_CURR' ,inplace=True)
    return data

def grant_credit(x):
    if (x>=0.5):
        return "accepted because the score is greater than 0.5"
    else:
        return "refused because the score is lower than 0.5"

data_load_state = st.text('Loading data...')
data = load_data(None)
# lgbm_optimise = joblib.load("lgbm.joblib")
# explainer = shap.Explainer(lgbm_optimise, data)
# shap_values = explainer(data)
# print("test shap values")
# print("*"*50)
# print(shap_values)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)    

liste_features = ["AMT_INCOME_TOTAL"]
with st.form("score_client"):
    id = st.selectbox("Select an Id from your dataset", data.index.sort_values().unique())
    submit = st.form_submit_button("submit")

    if submit:
        r = requests.post("https://scoringapi.herokuapp.com/predict", json = {"Id": id})
        parsed = json.loads(r.content)
        st.metric("Average score for the client id **{}** ".format(id), parsed["prediction"])
        fig = go.Figure()
        fig.add_trace(
            go.Indicator(mode="gauge+number", 
                         value=parsed["prediction"],
                         delta = {'reference': 160},
                         gauge = {'axis': {'range': [0, 1]}})
        )
        st.markdown("The credit claim for this client is therefore  **{}**".format(grant_credit(parsed["prediction"])))
        st.plotly_chart(fig, use_container_width=True)
        data_client =  data[data.index==id]
        st.subheader(" Here are some caracteristics of the client {}".format(id))
        st.markdown("The number of children of the client is **{}** \n and the total income is **{}$**".format(data_client["CNT_CHILDREN"].values[0],int(data_client["AMT_INCOME_TOTAL"].values[0])))
        st.sidebar.markdown('Cette section permet de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.')
        feature_to_compare = st.sidebar.selectbox('Quelle caractéristique souhaitez vous comparer', liste_features)
        st.markdown("""In comparison the average income is **{}$**""".format(int(data["AMT_INCOME_TOTAL"].mean())))




