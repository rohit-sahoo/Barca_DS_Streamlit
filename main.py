import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen

parser = Sbopen()

st.subheader("Here, we have La Liga data of all the Matches played by Barcelona in the Lionel Messi era")
df_competition = parser.competition()
df_laliga_matches = pd.DataFrame()


def getSeasonDict():    
    df_laliga_rows = df_competition[df_competition['competition_id'] == 11]
    season_name = df_laliga_rows['season_name']
    season_id = df_laliga_rows['season_id']
    season_dict = dict(zip(season_name, season_id))
    return season_dict

def getSelectedSeasonMatchData(selected_season):
    df_selected_season_rows = df_selected_season_rows[df_selected_season_rows['season_id'] == season_dict[selected_season]]

    # Iterate through the rows and fetch matches
    for index, row in df_selected_season_rows.iterrows():
        competition_id = row['competition_id']
        season_id = row['season_id']
        # Your logic or function based on 'competition_id' and 'season_id' values
        df_match = parser.match(competition_id=competition_id, season_id=season_id)
        # Append the matches to df_championsLeague_matches
        df_laliga_matches = df_laliga_matches.append(df_match, ignore_index=True)




season_dict = getSeasonDict()
selected_season = st.selectbox("Select the season to analyze", list(season_dict.keys()))
getSelectedSeasonMatchData(selected_season)

if selected_season:
    st.write(f"Analyzing season: {selected_season}")

    st.write("Match Data:")
    st.table(df_laliga_matches)