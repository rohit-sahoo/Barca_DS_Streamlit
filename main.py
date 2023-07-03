import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen

parser = Sbopen()

st.subheader("Here, we have La Liga data of all the Matches played by Barcelona in the Lionel Messi era")

def getSeasonDict():
    df_competition = parser.competition()
    df_laliga_rows = df_competition[df_competition['competition_id'] == 11]
    season_name = df_laliga_rows['season_name']
    season_id = df_laliga_rows['season_id']
    season_dict = dict(zip(season_name, season_id))
    return season_dict


season_dict = getSeasonDict()
selected_season = st.selectbox("Select the season to analyze", list(season_dict.keys()))


if selected_season:
    st.write(f"Analyzing season: {selected_season}")

    # Add your analysis code here based on the selected season
    # You can fetch and process the data for the selected season

    # Example: Display a table with the data
    data = {
        "Date": ["2021-01-01", "2021-01-05", "2021-01-10"],
        "Opponent": ["Real Madrid", "Atletico Madrid", "Sevilla"],
        "Result": ["W", "L", "D"],
    }

    st.write("Match Data:")
    st.table(data)