from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

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


def get_base_features(input_df):
    seed_everything(seed=SEED)
    output_df = input_df.copy()

    output_df['inning'] = 2 * (output_df['inning'].str[0].astype(int) - 1) + output_df['inning'].str.contains('裏')

    output_df['pitcherCommon'] = output_df['pitcher']
    output_df['batterCommon'] = output_df['batter']
    output_df.loc[~(output_df['pitcherCommon'].isin(tr_pitcher & ts_pitcher)), 'pitcherCommon'] = np.nan
    output_df.loc[~(output_df['batterCommon'].isin(tr_batter & ts_batter)), 'batterCommon'] = np.nan

    # label encoding
    cat_cols = output_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        f = output_df[col].notnull()
        output_df.loc[f, col] = LabelEncoder().fit_transform(output_df.loc[f, col].values)
        output_df.loc[~f, col] = -1
        output_df[col] = output_df[col].astype(int)
    
    output_df['inningHalf'] = output_df['inning'] % 2
    output_df['inningNumber'] = output_df['inning'] // 2
    output_df['outCount'] = output_df['inning'] * 3 + output_df['O']
    output_df['B_S_O'] = output_df['B'] + 4 * (output_df['S'] + 3 * output_df['O'])
    output_df['b1_b2_b3'] = output_df['b1'] * 1 + output_df['b2'] * 2 + output_df['b3'] * 4

    next_b = output_df.sort_values(['gameID', 'inning', 'O']).groupby(['gameID', 'inning'], group_keys=False)['b1', 'b2', 'b3'].shift(-1).rename(columns={'b1': 'n_b1', 'b2': 'n_b2', 'b3': 'n_b3'}) 
    output_df = pd.merge(output_df, next_b, left_index=True, right_index=True)

    def replace_b1(x):
        if pd.isnull(x['n_b1']):
            return x['b1']
        else:
            return x['n_b1']
    def replace_b2(x):
        if pd.isnull(x['n_b2']):
            return x['b2']
        else:
            return x['n_b2']
    def replace_b3(x):
        if pd.isnull(x['n_b3']):
            return x['b3']
        else:
            return x['n_b3']
    
    output_df['n_b1'] = output_df.apply(replace_b2, axis=1)
    output_df['n_b2'] = output_df.apply(replace_b2, axis=1)
    output_df['n_b3'] = output_df.apply(replace_b3, axis=1)
    
    output_df['plus_b1'] = output_df.apply(lambda x: x['b1'] < x['n_b1'], axis=1)
    output_df['plus_b2'] = output_df.apply(lambda x: x['b2'] < x['n_b2'], axis=1)
    output_df['plus_b3'] = output_df.apply(lambda x: x['b3'] < x['n_b3'], axis=1)
    
    output_df['minus_b1'] = output_df.apply(lambda x: x['b1'] > x['n_b1'], axis=1)
    output_df['minus_b2'] = output_df.apply(lambda x: x['b2'] > x['n_b2'], axis=1)
    output_df['minus_b3'] = output_df.apply(lambda x: x['b3'] > x['n_b3'], axis=1)
    
    return output_df

    
def aggregation(input_df, group_keys, group_values, agg_methods):
    new_df = []
    for agg_method in agg_methods:
        for col in group_values:
            if callable(agg_method):
                agg_method_name = agg_method.__name__
            else:
                agg_method_name = agg_method
            new_col = f'agg_{agg_method_name}_{col}_grpby_' + '_'.join(group_keys)
            agg_df = input_df[[col]+group_keys].groupby(group_keys)[[col]].agg(agg_method)
            agg_df.columns = [new_col]
            new_df.append(agg_df)
    new_df = pd.concat(new_df, axis=1).reset_index()

    output_df = pd.merge(input_df, new_df, on=group_keys, how='left')
    return output_df, list(new_df.columns)


def get_agg_gameID_inningHalf_features(input_df):
    group_keys = ['subGameID', 'inningHalf']
    group_values = ['S', 'B', 'b1', 'b2', 'b3']
    agg_methods = ['mean', 'std']
    output_df, cols = aggregation(
        input_df, group_keys=group_keys, group_values=group_values, agg_methods=agg_methods)
    return reduce_mem_usage(output_df)

    
'''
pivot table features
'''
def get_pivot_NMF9_features(input_df, n, value_col):
    pivot_df = pd.pivot_table(input_df, index='subGameID', columns='outCount', values=value_col, aggfunc=np.median)
    sc0 = MinMaxScaler().fit_transform(np.median(pivot_df.fillna(0).values.reshape(-1,54//3,3)[:,0::2,:], axis=-1))
    sc1 = MinMaxScaler().fit_transform(np.median(pivot_df.fillna(0).values.reshape(-1,54//3,3)[:,1::2,:], axis=-1))
    nmf = NMF(n_components=n, random_state=2021)
    nmf_df0 = pd.DataFrame(nmf.fit_transform(sc0), index=pivot_df.index).rename(
        columns=lambda x: f'pivot_{value_col}_NMF9T={x:02}')
    nmf_df1 = pd.DataFrame(nmf.fit_transform(sc1), index=pivot_df.index).rename(
        columns=lambda x: f'pivot_{value_col}_NMF9B={x:02}')
    nmf_df = pd.concat([nmf_df0, nmf_df1], axis=1)
    nmf_df = pd.merge(
        input_df, nmf_df, left_on='subGameID', right_index=True, how='left')
    return reduce_mem_usage(nmf_df)


# pivot tabel を用いた特徴量
def get_pivot_NMF27_features(input_df, n, value_col):
    pivot_df = pd.pivot_table(input_df, index='subGameID', columns='outCount', values=value_col, aggfunc=np.median)
    sc0 = MinMaxScaler().fit_transform(pivot_df.fillna(0).values.reshape(-1,54//3,3)[:,0::2].reshape(-1,27))
    sc1 = MinMaxScaler().fit_transform(pivot_df.fillna(0).values.reshape(-1,54//3,3)[:,1::2].reshape(-1,27))
    nmf = NMF(n_components=n, random_state=2021)
    nmf_df0 = pd.DataFrame(nmf.fit_transform(sc0), index=pivot_df.index).rename(
        columns=lambda x: f'pivot_{value_col}_NMF27T={x:02}')
    nmf_df1 = pd.DataFrame(nmf.fit_transform(sc1), index=pivot_df.index).rename(
        columns=lambda x: f'pivot_{value_col}_NMF27B={x:02}')
    nmf_df = pd.concat([nmf_df0, nmf_df1], axis=1)
    nmf_df = pd.merge(
        input_df, nmf_df, left_on='subGameID', right_index=True, how='left')
    return reduce_mem_usage(nmf_df)


# pivot tabel を用いた特徴量
def get_pivot_NMF54_features(input_df, n, value_col):
    pivot_df = pd.pivot_table(input_df, index='subGameID', columns='outCount', values=value_col, aggfunc=np.median)
    sc = MinMaxScaler().fit_transform(pivot_df.fillna(0).values)
    nmf = NMF(n_components=n, random_state=2021)
    nmf_df = pd.DataFrame(nmf.fit_transform(sc), index=pivot_df.index).rename(
        columns=lambda x: f'pivot_{value_col}_NMF54={x:02}')
    nmf_df = pd.merge(
        input_df, nmf_df, left_on='subGameID', right_index=True, how='left')
    return reduce_mem_usage(nmf_df)


'''
shift features
'''
def get_diff_feature(input_df, value_col, periods, in_inning=True, aggfunc=np.median):
pivot_df = pd.pivot_table(input_df, index='subGameID', columns='outCount', values=value_col, aggfunc=aggfunc)
if in_inning:
    dfs = []
    for inning in range(9):
        df0 = pivot_df.loc[:, [out+inning*6 for out in range(0,3)]].diff(periods, axis=1)
        df1 = pivot_df.loc[:, [out+inning*6 for out in range(3,6)]].diff(periods, axis=1)
        dfs += [df0, df1]
    pivot_df = pd.concat(dfs, axis=1).stack()
else:
    df0 = pivot_df.loc[:, [out+inning*6 for inning in range(9) for out in range(0,3)]].diff(periods, axis=1)
    df1 = pivot_df.loc[:, [out+inning*6 for inning in range(9) for out in range(3,6)]].diff(periods, axis=1)
    pivot_df = pd.concat([df0, df1], axis=1).stack()
return pivot_df


def get_shift_feature(input_df, value_col, periods, in_inning=True, aggfunc=np.median):
    pivot_df = pd.pivot_table(input_df, index='subGameID', columns='outCount', values=value_col, aggfunc=aggfunc)
    if in_inning:
        dfs = []
        for inning in range(9):
            df0 = pivot_df.loc[:, [out+inning*6 for out in range(0,3)]].shift(periods, axis=1)
            df1 = pivot_df.loc[:, [out+inning*6 for out in range(3,6)]].shift(periods, axis=1)
            dfs += [df0, df1]
        pivot_df = pd.concat(dfs, axis=1).stack()
    else:
        df0 = pivot_df.loc[:, [out+inning*6 for inning in range(9) for out in range(0,3)]].shift(periods, axis=1)
        df1 = pivot_df.loc[:, [out+inning*6 for inning in range(9) for out in range(3,6)]].shift(periods, axis=1)
        pivot_df = pd.concat([df0, df1], axis=1).stack()
    return pivot_df


def get_next_data(input_df, value_col, in_inning=True, nan_value=None):
    pivot_df = get_shift_feature(input_df, value_col, periods=-1, in_inning=in_inning)
    pivot_df.name = 'next_' + value_col
    output_df = pd.merge(
        input_df, pivot_df, left_on=['subGameID', 'outCount'], right_index=True, how='left')
    if nan_value is not None:
        output_df[pivot_df.name].fillna(nan_value, inplace=True)
    return output_df


def get_prev_data(input_df, value_col, in_inning=True, nan_value=None):
    pivot_df = get_shift_feature(input_df, value_col, periods=1, in_inning=in_inning)
    pivot_df.name = 'prev_' + value_col
    output_df = pd.merge(
        input_df, pivot_df, left_on=['subGameID', 'outCount'], right_index=True, how='left')
    if nan_value is not None:
        output_df[pivot_df.name].fillna(nan_value, inplace=True)
    return output_df
    

def get_next_diff(input_df, value_col, in_inning=True, nan_value=None):
    pivot_df = get_diff_feature(input_df, value_col, periods=-1, in_inning=in_inning)
    pivot_df.name = 'next_diff_' + value_col
    output_df = pd.merge(
        input_df, pivot_df, left_on=['subGameID', 'outCount'], right_index=True, how='left')
    if nan_value is not None:
        output_df[pivot_df.name].fillna(nan_value, inplace=True)
    return output_df


def get_prev_diff(input_df, value_col, in_inning=True, nan_value=None):
    pivot_df = get_diff_feature(input_df, value_col, periods=1, in_inning=in_inning)
    pivot_df.name = 'prev_diff_' + value_col
    output_df = pd.merge(
        input_df, pivot_df, left_on=['subGameID', 'outCount'], right_index=True, how='left')
    if nan_value is not None:
        output_df[pivot_df.name].fillna(nan_value, inplace=True)
    return output_df


'''
tf-idf features
'''
def get_tfidf(input_df, term_col, document_col):
    output_df = input_df.copy()
    output_df['dummy'] = 0
    tf1 = output_df[[document_col, term_col, 'dummy']].groupby([document_col, term_col])['dummy'].count()
    tf1.name = 'tf1'
    tf2 = output_df[[document_col, term_col, 'dummy']].groupby([document_col])['dummy'].count()
    tf2.name = 'tf2'
    idf1 = output_df[document_col].nunique()
    idf2 = output_df[[document_col, term_col, 'dummy']].groupby([term_col])[document_col].nunique()
    idf2.name = 'idf2'
    output_df = pd.merge(output_df, tf1, left_on=[document_col, term_col], right_index=True, how='left')
    output_df = pd.merge(output_df, tf2, left_on=[document_col], right_index=True, how='left')
    output_df['idf1'] = idf1
    output_df = pd.merge(output_df, idf2, left_on=[term_col], right_index=True, how='left')
    col_name = 'tfidf_' + term_col + '_in_' + document_col
    tf = np.log(1 + (1 + output_df['tf1']) / (1 + output_df['tf2']))
    idf = 1 + np.log((1 + output_df['idf1']) / (1 + output_df['idf2']))
    output_df[col_name] = tf * idf
    return output_df.drop(['tf1', 'tf2', 'idf1', 'idf2', 'dummy'], axis=1)


def get_skip(input_df):
    output_df = input_df.copy()

    next_skip_map = {}
    prev_skip_map = {}
    for key, group in output_df.groupby(['subGameID', 'inningHalf']):
        n = len(group)
        dist_map = {}
        batter = group.sort_values('outCount')['batter']
        for i in range(n - 1):
            b1 = batter.iloc[i]
            for d in range(1, 5):
                if i + d >= n:
                    break
                b2 = batter.iloc[i + d]

                if (b1, b2) in dist_map.keys():
                    if dist_map[(b1, b2)] < d:
                        dist_map[(b1, b2)] = d
                else:
                    dist_map[(b1, b2)] = d
            
        for i in range(len(batter) - 1):
            next_skip_map[batter.index[i]] = dist_map[(batter.iloc[i], batter.iloc[i+1])]
        for i in range(1, len(batter)):
            prev_skip_map[batter.index[i]] = dist_map[(batter.iloc[i-1], batter.iloc[i])]

    output_df['next_skip'] = output_df.index.map(next_skip_map).fillna(0).astype(np.int8)
    output_df['prev_skip'] = output_df.index.map(prev_skip_map).fillna(0).astype(np.int8)
    
    return output_df