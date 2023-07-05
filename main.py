import streamlit as st
import pandas as pd
from mplsoccer import Sbopen,Pitch
import matplotlib.pyplot as plt
import numpy as np

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
home_match = selected_opponent_matches[selected_opponent_matches['home_team_name'] == "Barcelona"]
home_matchID = home_match['match_id'].astype(int)

away_match = selected_opponent_matches[selected_opponent_matches['away_team_name'] == "Barcelona"]
away_matchID = away_match['match_id'].astype(int)



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
#home_events = selected_opponent_match_events[selected_opponent_match_events['match_id'].astype(int) == home_matchID]
#away_events = selected_opponent_match_events[selected_opponent_match_events['match_id'].astype(int) == away_matchID]


### 3. Ask user to select preferences like shot map, goal map etc.
analysis_name = ['Shots','Goals','Passes']
analysis_key = [1,2,3]
analysis_dict = dict(zip(analysis_name, analysis_key))
selected_analysis = st.selectbox("Select the technique to analyze", list(analysis_dict.keys()))

### 4. Creating plots for passes
completed_normal_passes = selected_opponent_match_events.loc[selected_opponent_match_events['type_name'] == 'Pass'].loc[selected_opponent_match_events['sub_type_name'].isna()].set_index('id')
#completed_normal_passes_home = home_events.loc[home_events['type_name'] == 'Pass'].loc[home_events['sub_type_name'].isna()].set_index('id')
#completed_normal_passes_away = away_events.loc[away_events['type_name'] == 'Pass'].loc[away_events['sub_type_name'].isna()].set_index('id')

def getPassesPerPlayerCount(df):
    player_passes = df.groupby('player_name').size().reset_index(name='total_passes')
    top_players = player_passes.nlargest(15, 'total_passes')
    # Create a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(top_players['player_name'], top_players['total_passes'])

    # Set the plot title and labels
    ax.set_title('Top 15 Players with Highest Passes')
    ax.set_xlabel('Player')
    ax.set_ylabel('Number of Passes')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)
    # Show the plot in Streamlit app
    st.pyplot(fig)

def getTeamPassingNetwork(df):
    df_pass = df[['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]
    #adjusting that only the surname of a player is presented.
    df_pass["player_name"] = df_pass["player_name"].apply(lambda x: str(x).split()[-1])
    df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])
    scatter_df = pd.DataFrame()
    for i, name in enumerate(df_pass["player_name"].unique()):
        passx = df_pass.loc[df_pass["player_name"] == name]["x"].to_numpy()
        recx = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_x"].to_numpy()
        passy = df_pass.loc[df_pass["player_name"] == name]["y"].to_numpy()
        recy = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_y"].to_numpy()
        scatter_df.at[i, "player_name"] = name
        #make sure that x and y location for each circle representing the player is the average of passes and receptions
        scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
        #calculate number of passes
        scatter_df.at[i, "no"] = df_pass.loc[df_pass["player_name"] == name].count().iloc[0]

    #adjust the size of a circle so that the player who made more passes
    scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)
    #counting passes between players
    df_pass["pair_key"] = df_pass.apply(lambda x: "_".join(sorted([x["player_name"], x["pass_recipient_name"]])), axis=1)
    lines_df = df_pass.groupby(["pair_key"]).x.count().reset_index()
    lines_df.rename({'x':'pass_count'}, axis='columns', inplace=True)
    #setting a treshold. You can try to investigate how it changes when you change it.
    lines_df = lines_df[lines_df['pass_count']>2]

    #plot once again pitch and vertices
    pitch = Pitch(line_color='grey')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, color='red', edgecolors='grey', linewidth=1, alpha=1, ax=ax["pitch"], zorder = 3)
    for i, row in scatter_df.iterrows():
        pitch.annotate(row.player_name, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax["pitch"], zorder = 4)

    for i, row in lines_df.iterrows():
            player1 = row["pair_key"].split("_")[0]
            player2 = row['pair_key'].split("_")[1]
            #take the average location of players to plot a line between them
            player1_x = scatter_df.loc[scatter_df["player_name"] == player1]['x'].iloc[0]
            player1_y = scatter_df.loc[scatter_df["player_name"] == player1]['y'].iloc[0]
            player2_x = scatter_df.loc[scatter_df["player_name"] == player2]['x'].iloc[0]
            player2_y = scatter_df.loc[scatter_df["player_name"] == player2]['y'].iloc[0]
            num_passes = row["pass_count"]
            #adjust the line width so that the more passes, the wider the line
            line_width = (num_passes / lines_df['pass_count'].max() * 10)
            #plot lines on the pitch
            pitch.lines(player1_x, player1_y, player2_x, player2_y,
                            alpha=1, lw=line_width, zorder=2, color="red", ax = ax["pitch"])

    fig.suptitle("Baracelona Passing Network against" + selected_opponent, fontsize = 30)
    #st.pyplot(fig)
    plt.show()




def getPassingNetwork(df):
    ## for Barcelona
    barca_passes = df[df['team_name'] == 'Barcelona']
    getTeamPassingNetwork(barca_passes)

    ## for selected opponent
    #opponent_passes = df[df['team_name'] == selected_opponent]
    #getTeamPassingNetwork(opponent_passes)




### 5. Creating plots for Shots


### 6. Creating plots for goals

if selected_season:
    st.write(f"Analyzing season: {selected_season}")

if selected_opponent:
    st.write(f"Analyzing opponent: {selected_opponent}")

if selected_analysis:
    st.write(f"Analyzing : {selected_analysis}")
    if selected_analysis == "Passes":
        st.write(f"Analyzing total passes for both home and away games: {selected_analysis}")
        getPassesPerPlayerCount(completed_normal_passes)

        st.write("Home Match ID", home_matchID)
        st.write("Match Id type", type(home_matchID))
        st.write("Away match ID", away_matchID)

        #st.write(f"Analyzing total passes for home game: {selected_analysis}")
        #getPassesPerPlayerCount(home_events)
        

        #st.write(f"Analyzing total passes for away game: {selected_analysis}")
        #getPassesPerPlayerCount(away_events)
        




        #getPassingNetwork(completed_normal_passes)

