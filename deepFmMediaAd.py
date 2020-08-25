import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr.feature_column import SparseFeat,get_feature_names
from deepctr.models import DeepFM, xDeepFM

if __name__ == "__main__":
    data = pd.read_csv('./datasets/refined_data_10K.csv')
    data['day'] = data['time'].apply(lambda x: str(x)[4:6])
    data['hour'] = data['time'].apply(lambda x: str(x)[6:])

    sparse_features = ['time', 'device_os', 'creative', 'publisher', 'campaign',
                       'publisher_category', 'source', 'app_domain', 'app_category', 'device_id',
                       'device_model', 'city', 'country',
                       'browser', 'user_score', 'user_value', 'md_1', 'md_2', 'md_3', ]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    target = ['click']

    # 1. Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name

    field_info = dict(browser='user', user_score='user', user_value='user',
                      md_1='user', md_2='user', md_3='user', device_os='user',
                      creative='context', publisher='context', publisher_category='context',
                      source='item', app_domain='item', app_category='item',
                      device_model='user', city='user', country='user', time='context',
                      device_id='user', campaign='context')

    fixlen_feature_columns = [
        SparseFeat(name, vocabulary_size=data[name].nunique(), embedding_dim=16, use_hash=False, dtype='int32',
                   group_name=field_info[name]) for name in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=0.5, dnn_hidden_units=(256, 256))
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans, eps=1e-7), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
