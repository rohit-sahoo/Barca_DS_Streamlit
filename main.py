import streamlit as st
import pandas as pd
from mplsoccer import Sbopen,Pitch
import matplotlib.pyplot as plt
import numpy as np

parser = Sbopen()
team = "Barcelona"

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
home_matchID = int(home_match['match_id'].item())

away_match = selected_opponent_matches[selected_opponent_matches['away_team_name'] == "Barcelona"]
away_matchID = int(away_match['match_id'].item())



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
home_events = selected_opponent_match_events[selected_opponent_match_events['match_id'] == home_matchID]
away_events = selected_opponent_match_events[selected_opponent_match_events['match_id'] == away_matchID]


### 3. Ask user to select preferences like shot map, goal map etc.
analysis_name = ['Shots','Goals','Passes']
analysis_key = [1,2,3]
analysis_dict = dict(zip(analysis_name, analysis_key))
selected_analysis = st.selectbox("Select the technique to analyze", list(analysis_dict.keys()))

### 4. Creating plots for passes
completed_normal_passes = selected_opponent_match_events.loc[selected_opponent_match_events['type_name'] == 'Pass'].loc[selected_opponent_match_events['sub_type_name'].isna()].set_index('id')
completed_normal_passes_home = home_events.loc[home_events['type_name'] == 'Pass'].loc[home_events['sub_type_name'].isna()].set_index('id')
completed_normal_passes_away = away_events.loc[away_events['type_name'] == 'Pass'].loc[away_events['sub_type_name'].isna()].set_index('id')

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
    st.pyplot(fig)
    #plt.show()

def getSubstitutionsEvent(df):
    sub = df.loc[df["type_name"] == "Substitution"].loc[df["team_name"] == "Barcelona"].loc[df["match_id"] == df['match_id']].iloc[0]["index"]
    #make df with successfull passes by England until the first substitution
    mask_england = (df.type_name == 'Pass') & (df.team_name == "Barcelona") & (df.index < sub) & (df.outcome_name.isnull()) & (df.sub_type_name != "Throw-in")
    #taking necessary columns
    df_pass = df.loc[mask_england, ['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]
    #adjusting that only the surname of a player is presented.
    df_pass["player_name"] = df_pass["player_name"].apply(lambda x: str(x).split()[-1])
    df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])
    return df_pass

def getPassingNetwork(df):

    df_subs = getSubstitutionsEvent(df)
    getTeamPassingNetwork(df_subs)

    ## for selected opponent
    #opponent_passes = df[df['team_name'] == selected_opponent]
    #getTeamPassingNetwork(opponent_passes)


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if (val == value).any():
            return key
    return None

def getPlayersForMatch(matchId_list):

    df_barca_players = dict()

    for matchId in matchId_list:
        df_lineup = parser.lineup(matchId)
        df_lineup = df_lineup[df_lineup['team_name'] == "Barcelona"]
        p_name = df_lineup['player_name']
        p_nickname = df_lineup['player_nickname']
        player_dict = dict(zip(p_nickname, p_name))
        df_barca_players.update(player_dict)
        #print(df_barca_players)

    return df_barca_players


players_dict = getPlayersForMatch(selected_opponent_matchID)
selected_player= st.selectbox("Select the player to analyze", list(players_dict.keys()))

def getPlayersPassesPlot(df):
    #passes = df.loc[df['type_name'] == 'Pass'].loc[df['sub_type_name'] != 'Throw-in'].set_index('id')
    mask_bronze = (df.type_name == 'Pass') & (df.player_name == players_dict[selected_player])
    df_pass = df.loc[mask_bronze, ['x', 'y', 'end_x', 'end_y']]

    if len(df_pass)>0:
        pitch = Pitch(line_color='black')
        fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                            endnote_height=0.04, title_space=0, endnote_space=0)
        pitch.arrows(df_pass.x, df_pass.y,
                    df_pass.end_x, df_pass.end_y, color = "blue", ax=ax['pitch'])
        pitch.scatter(df_pass.x, df_pass.y, alpha = 0.2, s = 500, color = "blue", ax=ax['pitch'])
        fig.suptitle(f"{selected_player} passes against {selected_opponent}", fontsize = 30)
        st.pyplot(fig)

    else:
        st.write("Player had zero passes throughout the game, this could also mean that player was not substituted in the match or was not in the starting eleven")

def getPassingHeatMap(df):
    #declare an empty dataframe
    danger_passes = pd.DataFrame()
    for period in [1, 2]:
        #keep only accurate passes by England that were not set pieces in this period
        mask_pass = (df.team_name == team) & (df.type_name == "Pass") & (df.outcome_name.isnull()) & (df.period == period) & (df.sub_type_name.isnull())
        #keep only necessary columns
        passes = df.loc[mask_pass, ["x", "y", "end_x", "end_y", "minute", "second", "player_name"]]
        #keep only Shots by England in this period
        mask_shot = (df.team_name == team) & (df.type_name == "Shot") & (df.period == period)
        #keep only necessary columns
        shots = df.loc[mask_shot, ["minute", "second"]]
        #convert time to seconds
        shot_times = shots['minute']*60+shots['second']
        shot_window = 15
        #find starts of the window
        shot_start = shot_times - shot_window
        #condition to avoid negative shot starts
        shot_start = shot_start.apply(lambda i: i if i>0 else (period-1)*45)
        #convert to seconds
        pass_times = passes['minute']*60+passes['second']
        #check if pass is in any of the windows for this half
        pass_to_shot = pass_times.apply(lambda x: True in ((shot_start < x) & (x < shot_times)).unique())

        #keep only danger passes
        danger_passes_period = passes.loc[pass_to_shot]
        #concatenate dataframe with a previous one to keep danger passes from the whole tournament
        danger_passes = pd.concat([danger_passes, danger_passes_period], ignore_index = True)


    #plot vertical pitch
    pitch = Pitch(line_zorder=2, line_color='black')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #get the 2D histogram
    bin_statistic = pitch.bin_statistic(danger_passes.x, danger_passes.y, statistic='count', bins=(6, 5), normalize=False)
    #normalize by number of games
    bin_statistic["statistic"] = bin_statistic["statistic"]
    #make a heatmap
    pcm  = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
    #legend to our plot
    ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
    cbar = plt.colorbar(pcm, cax=ax_cbar)
    fig.suptitle('Danger passes by ' + team , fontsize = 30)
    st.pyplot(fig)

    return danger_passes

def plotDangerousPlayerPlots(df):

    # Count passes by player and normalize them
    pass_count = df.groupby("player_name").size().reset_index(name="pass_count")

    # Create a bar plot
    fig, ax = plt.subplots()
    ax.bar(pass_count["player_name"], pass_count["pass_count"])
    
    # Set plot title and labels
    ax.set_title("Passes by Player")
    ax.set_xlabel("Player")
    ax.set_ylabel("Key Passes Count")

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Show the plot in Streamlit app
    st.pyplot(fig)


### 5. Creating plots for Shots

def plotShots(df):
    #create pitch
    pitch = Pitch(line_color='black')
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                        endnote_height=0.04, title_space=0, endnote_space=0)
    #query
    mask_barca = (df.type_name == 'Shot') & (df.team_name == team)
    #finding rows in the df and keeping only necessary columns
    df_barca = df.loc[mask_barca, ['x', 'y', 'outcome_name', "player_name"]]

    #plot them - if shot ended with Goal - alpha 1 and add name
    #for Barcelona
    for i, row in df_barca.iterrows():
        if row["outcome_name"] == 'Goal':
            pitch.scatter(row.x, row.y, alpha = 1, s = 500, color = "red", ax=ax['pitch'])
            pitch.annotate(row["player_name"], (row.x + 1, row.y - 2), ax=ax['pitch'], fontsize = 12)
        else:
            pitch.scatter(row.x, row.y, alpha = 0.2, s = 500, color = "red", ax=ax['pitch'])

    mask_opponent = (df.type_name == 'Shot') & (df.team_name != team)
    df_opponent = df.loc[mask_opponent, ['x', 'y', 'outcome_name', "player_name"]]

    #for opponent we need to revert coordinates
    for i, row in df_opponent.iterrows():
        if row["outcome_name"] == 'Goal':
            pitch.scatter(120 - row.x, 80 - row.y, alpha = 1, s = 500, color = "blue", ax=ax['pitch'])
            pitch.annotate(row["player_name"], (120 - row.x + 1, 80 - row.y - 2), ax=ax['pitch'], fontsize = 12)
        else:
            pitch.scatter(120 - row.x, 80 - row.y, alpha = 0.2, s = 500, color = "blue", ax=ax['pitch'])


    fig.suptitle(f"{team} (red) and {selected_opponent} (blue) shots", fontsize = 30)
    st.pyplot(fig)

def plotShotsBarPlot(df):
    # Group by 'player_name' and calculate the sum of 'shots' for each player
    player_shots = df.groupby('player_name')['shots'].sum().reset_index()

    # Sort the DataFrame by 'shots' column in descending order
    player_shots_sorted = player_shots.sort_values('shots', ascending=False)

    # Create a bar chart using Streamlit
    st.bar_chart(player_shots_sorted.set_index('player_name'))


### 6. Creating plots for goals

if selected_season:
    st.write(f"Analyzing season: {selected_season}")

if selected_opponent:
    st.write(f"Analyzing opponent: {selected_opponent}")

if selected_analysis:
    st.write(f"Analyzing : {selected_analysis}")
    if selected_analysis == "Passes":

        if selected_player:
            st.write(f"Analyzing passes for {selected_player} against {selected_opponent}")
            st.write("1. Home Game")
            getPlayersPassesPlot(home_events)
            st.write("2. Away Game")
            getPlayersPassesPlot(away_events)

        st.write(f"Analyzing total passes for both home and away games:")
        getPassesPerPlayerCount(completed_normal_passes)

        st.write(f"Analyzing total passes for home game:")
        getPassesPerPlayerCount(home_events)
        
        st.write(f"Analyzing total passes for away game:")
        getPassesPerPlayerCount(away_events)
        
        st.write(f"Passing network of Barcelona against: {selected_opponent}")
        getPassingNetwork(selected_opponent_match_events)

        st.write("Passing heatmap - Most dangerous passes heatmap at home")
        df_dangerPasses_home = getPassingHeatMap(home_events)

        st.write("Passing heatmap - Most dangerous passes heatmap at away")
        df_dangerPasses_away = getPassingHeatMap(away_events)

        st.write("Most dangerous passes bar plot at home")
        plotDangerousPlayerPlots(df_dangerPasses_home)

        st.write("Most dangerous passes bar plot at away")
        plotDangerousPlayerPlots(df_dangerPasses_away)

        #st.write("Passing probabilities that lead to a shot")
        #getPassingProbability()

    if selected_analysis == "Shots":
        
        st.write("Plotting shots for the match at home")
        plotShots(home_events)

        st.write("Plotting shots for the match at away")
        plotShots(away_events)

        st.write("Shots Barplot for the season")
        plotShotsBarPlot(selected_opponent_match_events)

        



