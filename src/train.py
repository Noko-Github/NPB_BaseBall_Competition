def random_sampling(input_df, n_sample=10):
    dfs = []
    tr_df = input_df[input_df['y'].notnull()].copy()
    ts_df = input_df[input_df['y'].isnull()].copy()
    for i in tqdm(range(n_sample)):
        df = tr_df.groupby(['gameID', 'outCount']).apply(lambda x: x.sample(n=1, random_state=i)).reset_index(drop=True)
        df['subGameID'] = df['gameID'] * n_sample + i
        dfs.append(df)
    ts_df['subGameID'] = ts_df['gameID'] * n_sample
    return pd.concat(dfs + [ts_df], axis=0