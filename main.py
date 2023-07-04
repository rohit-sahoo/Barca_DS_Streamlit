import streamlit as st
import pandas as pd
from mplsoccer import Sbopen
import matplotlib.pyplot as plt

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
### 1. Filtering matches of selected opponents
def getSelectedOpponentMatches(selected_opponent):
    selected_matches = df_selected_matches[(df_selected_matches['home_team_name'] == selected_opponent) | (df_selected_matches['away_team_name'] == selected_opponent)]
    match_ids = selected_matches['match_id'].tolist()
    df_opponent_matches = selected_matches
    return df_opponent_matches,match_ids

selected_opponent_matches,selected_opponent_matchID = getSelectedOpponentMatches(selected_opponent)

### 2. Fetching match events with selected opponents
# Function to fetch matches based on 'competition_id' and 'season_id' values
def fetch_events(row):
    # Your logic or function based on 'competition_id' and 'season_id' values
    df_event = parser.event(match_id=row)
    return df_event[0]

def getSelectedOpponentMatchEvents(selected_opponent_matchID):
    df_laliga_events = pd.DataFrame()
    # Iterate through the rows and fetch matches
    for row in selected_opponent_matchID:
        events = fetch_events(row)
        # Append the matches to df_championsLeague_matches
        df_laliga_events = pd.concat([df_laliga_events, events], ignore_index=True)
    return df_laliga_events


selected_opponent_match_events = getSelectedOpponentMatchEvents(selected_opponent_matchID)
home_events = selected_opponent_match_events[selected_opponent_match_events['home_team_name'] == 'Barcelona']
away_events = selected_opponent_match_events[selected_opponent_match_events['away_team_name'] == 'Barcelona']


### 3. Ask user to select preferences like shot map, goal map etc.
analysis_name = ['Shots','Goals','Passes']
analysis_key = [1,2,3]
analysis_dict = dict(zip(analysis_name, analysis_key))
selected_analysis = st.selectbox("Select the technique to analyze", list(analysis_dict.keys()))

### 4. Creating plots for passes
completed_normal_passes = selected_opponent_match_events.loc[selected_opponent_match_events['type_name'] == 'Pass'].loc[selected_opponent_match_events['sub_type_name'].isna()].set_index('id')
#completed_normal_passes_home = completed_normal_passes[completed_normal_passes['home_team_name'] == 'Barcelona']
#completed_normal_passes_away = completed_normal_passes[completed_normal_passes['away_team_name'] == 'Barcelona']

def getPassesPerPlayerCount(df):
    player_passes = df.groupby('player_name').size().reset_index(name='total_passes')
    top_players = player_passes.nlargest(15, 'total_passes')
    # Create a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(top_players['player_name'], top_players['passes'])

    # Set the plot title and labels
    ax.set_title('Top 15 Players with Highest Passes')
    ax.set_xlabel('Player')
    ax.set_ylabel('Number of Passes')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Show the plot in Streamlit app
    st.pyplot(fig)



### 5. Creating plots for Shots


### 6. Creating plots for goals

if selected_season:
    st.write(f"Analyzing season: {selected_season}")

if selected_opponent:
    st.write(f"Analyzing opponent: {selected_opponent}")
    st.write("Match Data:")
    st.table(selected_opponent_matches)

if selected_analysis:
    st.write(f"Analyzing : {selected_analysis}")
    if selected_analysis == "Passes":
        getPassesPerPlayerCount(completed_normal_passes)

