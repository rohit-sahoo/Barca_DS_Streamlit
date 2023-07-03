import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen

parser = Sbopen()

st.subheader("Here, we have La Liga data of all the Matches played by Barcelona in the Lionel Messi era")
df_competition = parser.competition()


def getSeasonDict():    
    df_laliga_rows = df_competition[df_competition['competition_id'] == 11]
    season_name = df_laliga_rows['season_name']
    season_id = df_laliga_rows['season_id']
    season_dict = dict(zip(season_name, season_id))
    return season_dict

# Function to fetch matches based on 'competition_id' and 'season_id' values
def fetch_matches(row):
    competition_id = row['competition_id']
    season_id = row['season_id']
    # Your logic or function based on 'competition_id' and 'season_id' values
    df_match = parser.match(competition_id=competition_id, season_id=season_id)
    return df_match


def getSelectedSeasonMatchData(selected_season):
    desired_season = season_dict[selected_season]
    df_laliga_rows = df_competition[(df_competition['competition_id'] == 11) & (df_competition['season_id'] == desired_season)]
    
    # Iterate through the rows and fetch matches
    for index, row in df_laliga_rows.iterrows():
        matches = fetch_matches(row)
        # Append the matches to df_championsLeague_matches
        selectedMatches = selectedMatches.append(matches, ignore_index=True)

df_selected_season_matches = pd.DataFrame()
selectedMatches = pd.DataFrame()

season_dict = getSeasonDict()
selected_season = st.selectbox("Select the season to analyze", list(season_dict.keys()))
df_selected_season_matches =  getSelectedSeasonMatchData(selected_season)


if selected_season:
    st.write(f"Analyzing season: {selected_season}")

    st.write("Match Data:")
    st.table(df_selected_season_matches)