
def add_team_postfix(input_df):
    output_df = input_df.copy()

    top = output_df['inning'].str.contains('表')
    output_df.loc[top, 'batter'] = output_df.loc[top, 'batter'] + '@' + output_df.loc[top, 'topTeam'].astype(str)
    output_df.loc[~top, 'batter'] = output_df.loc[~top, 'batter'] + '@' + output_df.loc[~top, 'bottomTeam'].astype(str)
    output_df.loc[ top, 'pitcher'] = output_df.loc[ top, 'pitcher'] + '@' + output_df.loc[ top, 'bottomTeam'].astype(str)
    output_df.loc[~top, 'pitcher'] = output_df.loc[~top, 'pitcher'] + '@' + output_df.loc[~top, 'topTeam'].astype(str)

    return output_df


def add_batter_order(input_df, is_train=True):
    pass 
    
    
def fill_na(input_df):
    output_df = input_df.copy()

    output_df['pitcherHand'] = output_df['pitcherHand'].fillna('R')
    output_df['batterHand'] = output_df['batterHand'].fillna('R')
    output_df['pitchType'] = output_df['pitchType'].fillna('-')
    output_df['speed'] = output_df['speed'].str.extract(r'(\d+)').fillna(method='ffill')
    output_df['ballPositionLabel'] = output_df['ballPositionLabel'].fillna('中心')
    output_df['ballX'] = output_df['ballX'].fillna(0).astype(int)
    output_df['ballY'] = output_df['ballY'].map({chr(ord('A')+i):i+1 for i in range(11)})
    output_df['ballY'] = output_df['ballY'].fillna(0).astype(int)
    output_df['dir'] = output_df['ballY'].map({chr(ord('A')+i):i+1 for i in range(26)})
    output_df['dir'] = output_df['dir'].fillna(0).astype(int)
    output_df['dist'] = output_df['dist'].fillna(0)
    output_df['battingType'] = output_df['battingType'].fillna('G')
    output_df['isOuts'] = output_df['isOuts'].fillna('-1').astype(int)

    return output_df

    