
#ニューラルネットを作成する関数定義
def create_model_NN(activation, n_layers, n_neurons, solver):
    hidden_layer_sizes=[]
    
    #与えられたパラメータのレイヤを作成
    for i in range(n_layers):
        hidden_layer_sizes.append(n_neurons[i])
    #print('hidden_layer_sizes -> ' + str(hidden_layer_sizes))
    
    #ニューラルネットのモデルを作成
    model = MLPRegressor(activation = activation,
                         hidden_layer_sizes=hidden_layer_sizes,
                         solver = solver,
                         random_state=42
                        )
    #標準化とニューラルネットのパイプラインを作成
    pipe = make_pipeline(StandardScaler(),model)
    return pip

# テストデータの「Speed」を予測する関数
def pred_speed_of_test_data(train_x,test,target_speed,param):
    ###################################
    ### パラメータの設定
    ##################################
    activation = param['activation']
    n_layers = param['n_layers']
    n_neurons=[]
    for i in range(n_layers):
        n_neurons.append(param['neuron' + str(i).zfill(2)])
    solver = param['solver']
    
    ###################################
    ### CVの設定
    ##################################
    
    FOLD_NUM = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    scores = []
    mlp_pred = 0

    for i, (tdx, vdx) in enumerate(kf.split(X=train_x)):
        X_train, X_valid, y_train, y_valid = train_x.iloc[tdx], train_x.iloc[vdx], target_speed[tdx], target_speed[vdx]
        #モデルを作成
        mlp  = create_model_NN(activation, n_layers, n_neurons, solver)
        # 学習
        mlp.fit(X_train,y_train)
        # 予測
        mlp_pred += mlp.predict(test) / FOLD_NUM

    print('#######################################################')
    print('### Seed was predicted #######')
    print('#######################################################')
    return mlp_pred


def predict(input_df, train_length)
    X_train = input_df.iloc[:train_length].drop('y', axis=1)
    X_test = input_df.iloc[train_length:].reset_index(drop=True).drop('y', axis=1)

    # Speed予測用のハイパーパラメータ
    param = {
    "activation": 'tanh',
    "n_layers": 9,
    "neuron00": 45,
    "neuron01": 52,
    "neuron02": 57,
    "neuron03": 79,
    "neuron04": 21,
    "neuron05": 102,
    "neuron06": 118,
    "neuron07": 31,
    "neuron08": 66,
    "solver": 'sgd',
    }

    target_speed = input_df.iloc[:train_length]['speed']

    ## NNの予測
    speed_pred = pred_speed_of_test_data(X_train,X_test,target_speed,param)