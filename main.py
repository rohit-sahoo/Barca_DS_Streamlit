import streamlit as st
import pandas as pd
from mplsoccer import Sbopen

parser = Sbopen()

st.subheader("Here, we have La Liga data of all the Matches played by Barcelona in the Lionel Messi era")
df_competition = parser.competition()

def getSeasonDict():
    df_laliga_rows = df_competition[df_competition['competition_id'] == 11]
    season_name = df_laliga_rows['season_name']
    season_id = df_laliga_rows['season_id']
    season_dict = dict(zip(season_name, season_id))
    return season_dict

def getSelectedSeasonMatchData(selected_season):
    df_selected_season_rows = df_competition[(df_competition['competition_id'] == 11) & (df_competition['season_id'] == season_dict[selected_season])]
    df_laliga_matches = pd.DataFrame()

    # Iterate through the rows and fetch matches
    for index, row in df_selected_season_rows.iterrows():
        current_competition_id = row['competition_id']
        current_season_id = row['season_id']
        # Your logic or function based on 'competition_id' and 'season_id' values
        df_match = parser.match(competition_id=current_competition_id, season_id=current_season_id)
        df_laliga_matches = pd.concat([df_laliga_matches, df_match], ignore_index=True)
    
    return df_laliga_matches

season_dict = getSeasonDict()
selected_season = st.selectbox("Select the season to analyze", list(season_dict.keys()))
df_selected_matches = getSelectedSeasonMatchData(selected_season)

## now that we have all the matches data, we have to filter out the opponents for the selected season
def getSelectedSeasonOpponents(selected_season):
    opponents = []
    for index, row in df_selected_matches.iterrows():
        if row['home_team_name'] == 'Barcelona':
            opponents.append(row['away_team_name'])
        elif row['away_team_name'] == 'Barcelona':
            opponents.append(row['home_team_name'])

    unique_opponents = set(opponents)
    unique_opponents_list = list(unique_opponents)
    return unique_opponents_list

opponents_dict = getSelectedSeasonOpponents(selected_season)
selected_opponent = st.selectbox("Select the opponent to analyze", opponents_dict)

## after selecting the opponent, we have to analyze Barcelona's performance against selected team in home and away games
def getSelectedOpponentMatches(selected_opponent):
    selected_matches = df_selected_matches[(df_selected_matches['home_team_name'] == selected_opponent) | (df_selected_matches['away_team_name'] == selected_opponent)]
    match_ids = selected_matches['match_id'].tolist()
    df_opponent_matches = selected_matches
    return df_opponent_matches

selected_opponent_matches = getSelectedOpponentMatches(selected_opponent)


if selected_season:
    st.write(f"Analyzing season: {selected_season}")
    st.write("Match Data:")
    st.table(df_selected_matches)


if selected_opponent:
    st.write(f"Analyzing opponent: {selected_opponent}")
    st.write("Match Data:")
    st.table(selected_opponent_matches)
