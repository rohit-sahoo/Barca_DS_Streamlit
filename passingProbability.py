from itertools import combinations_with_replacement
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost

# instantiate a parser object
parser = Sbopen()

df = parser.event(3773672)[0]


def plotPasseswithShotEnd(df,unique_possessions,passes):

    for i in unique_possessions:
        # plot possession chain that ended with shot
        chain = df.loc[df["possession"] == i]
        # get passes
        passes_in = passes.loc[df["possession"] == i]
        # get events different than pass
        not_pass = chain.loc[chain["type_name"] != "Pass"].iloc[:-1]
        # shot is the last event of the chain (or should be)
        shot = chain.iloc[-1]
        shot_taken_by = shot['player_name']
        shot_outcome = shot['outcome_name']
        if shot_outcome == 'Off T':
            shot_outcome = "Off Target shot"
        elif shot_outcome == 'Saved':
            shot_outcome = 'Saved Shot'
        elif shot_outcome == 'Blocked':
            shot_outcome = 'Blocked Shot'

        # plot
        pitch = Pitch(line_color='black', pitch_type='custom', pitch_length=120, pitch_width=80, line_zorder=2)
        fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                             endnote_height=0.04, title_space=0, endnote_space=0)
        # passes
        max_value = passes_in["xT"].max()  # Maximum xT value among passes
        pitch.arrows(passes_in.x0, passes_in.y0, passes_in.x1, passes_in.y1, color="blue", ax=ax['pitch'], zorder=3)
        
        # Annotate arrows with xT values
        for i, row in passes_in.iterrows():
            xT_value = row["xT"]
            arrow_x = (row.x0 + row.x1) / 2
            arrow_y = (row.y0 + row.y1) / 2
            size = 15 * (xT_value / max_value)  # Adjust the size based on xT value
            ax['pitch'].text(arrow_x, arrow_y, f"{xT_value:.3f}", color="Red", ha='center', va='center',fontsize = size)

        
        
        # shot
        pitch.arrows(shot.x0, shot.y0, shot.x1, shot.y1, color="red", ax=ax['pitch'], zorder=3)
        # other passes like arrows
        pitch.lines(not_pass.x0, not_pass.y0, not_pass.x1, not_pass.y1, color="grey", lw=1.5, ls='dotted', ax=ax['pitch'])
        ax['title'].text(0.5, 0.5, f'Passes leading to a {shot_outcome} taken by {shot_taken_by}', ha='center', va='center', fontsize=20)
        print("plottiung plot")
        plt.show()


def passingProbabilityPlots(df):

    unique_possessions = df[df['type_name'] == 'Shot']['possession'].unique()
    df_filtered = pd.DataFrame()

    for possession in unique_possessions:
        first_row_index = df.loc[df['possession'] == possession].index[0]
        index_shot = df.loc[(df['possession'] == possession) & (df['type_name'] == 'Shot')].index
        if len(index_shot) > 0:
            df_copy = df.copy()
            df_selected = df_copy[first_row_index:index_shot.item() + 1]
            df_filtered = pd.concat([df_filtered, df_selected], ignore_index=True)

    df_filtered.reset_index(drop=True, inplace=True)


    df_filtered2 = df_filtered.copy()

    for possession in unique_possessions:
        possession_mask = df_filtered2['possession'] == possession
        shot_xg = df_filtered2.loc[possession_mask & (df_filtered2['type_name'] == 'Shot'), 'shot_statsbomb_xg'].values[0]
        df_filtered2.loc[possession_mask, 'xG'] = shot_xg

    df2 = df.copy()
    df2 = df2[~df2['possession'].isin(unique_possessions)]
    df2 = pd.concat([df2, df_filtered2], ignore_index=True)

    df2['shot_end'] = df2['possession'].isin(unique_possessions).astype(int)

    df3 = df2.copy()

    #columns with coordinates
    df3["x0"] = df3['x']
    df3["c0"] = abs(40 - df2['y']) 
    df3["x1"] = df3['end_x']
    df3["c1"] = abs(40 - df2['end_y']) 


    #for plotting
    df3["y0"] = df3['y']
    df3["y1"] = df3['end_y']


    #model variables
    var = ["x0", "x1", "c0", "c1"]

    #combinations
    inputs = []
    #one variable combinations
    inputs.extend(combinations_with_replacement(var, 1))
    #2 variable combinations
    inputs.extend(combinations_with_replacement(var, 2))
    #3 variable combinations
    inputs.extend(combinations_with_replacement(var, 3))

    #make new columns
    for i in inputs:
        #columns length 1 already exist
        if len(i) > 1:
            #column name
            column = ''
            x = 1
            for c in i:
                #add column name to be x0x1c0 for example
                column += c
                #multiply values in column
                x = x*df3[c]
            #create a new column in df
            df3[column] = x
            #add column to model variables
            var.append(column)



    passes = df3.loc[df3["type_name"].isin(["Pass"])]

    X = passes[var].values 
    y = passes["shot_end"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123, stratify = y)
    model = xgboost.XGBClassifier(n_estimators = 100, ccp_alpha=0, max_depth=4, min_samples_leaf=10,
                        random_state=123)


    scores = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, n_jobs = -1)
    #print(np.mean(scores), np.std(scores))
    model.fit(X_train, y_train)
    #print(model.score(X_train, y_train))
    y_pred = model.predict(X_test)
    #print(model.score(X_test, y_test))


    #predict if ended with shot
    passes = df3.loc[df3["type_name"].isin(["Pass"])]
    X = passes[var].values
    y = passes["shot_end"].values

    #predict probability of shot ended
    y_pred_proba = model.predict_proba(X)[::,1]

    passes["shot_prob"] = y_pred_proba
    #OLS
    shot_ended = passes.loc[passes["shot_end"] == 1]
    X2 = shot_ended[var].values
    y2 = shot_ended["xG"].values
    lr = LinearRegression()
    lr.fit(X2, y2)
    y_pred = lr.predict(X)
    passes["xG_pred"] = y_pred
    #calculate xGchain
    passes["xT"] = passes["xG_pred"]*passes["shot_prob"]

    passes[["xG_pred", "shot_prob", "xT"]].head(5)
    plotPasseswithShotEnd(df3,unique_possessions,passes)

passingProbabilityPlots(df)