import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import keras.backend as K
from keras.optimizers import Adam
from keras import regularizers
from keras.models import Model
from sklearn import metrics
from keras.regularizers import l2
from keras.layers import *
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

def format_predict_y(predict_y):

    predict_label=[]
    predict_score=[]
    for y_score in predict_y:
        predict_score.append(y_score[0])
        if y_score[0]>y_score[1]:
            predict_label.append(1)
        else:
            predict_label.append(0)

    return np.array(predict_label), predict_score

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def get_csv_sample(csv_sample_file, positive_sample_size):
    data = np.array(pd.read_csv(csv_sample_file))#inputfile
    X1 = data[0:positive_sample_size, 1:]
    Y1 = data[0:positive_sample_size, 0]
    X2 = data[positive_sample_size:, 1:]
    Y2 = data[positive_sample_size:, 0]
    X = np.concatenate([X1, X2], 0)
    Y = np.concatenate([Y1, Y2], 0)

    return X, Y

def get_acidNum_csv(posi_sample_csv, nega_sample_csv):
    posi_data = np.array(pd.read_csv(posi_sample_csv))
    posi_X = posi_data[0:, 1:]
    posi_Y = posi_data[0:, 0]
    nega_data = np.array(pd.read_csv(nega_sample_csv))
    nega_X = nega_data[0:, 1:]
    nega_Y = nega_data[0:, 0]

    X = np.concatenate([posi_X, nega_X], 0)
    Y = np.concatenate([posi_Y, nega_Y], 0)

    return X, Y

def get_RRC_matrix_csv(RRC_matrix_csv, pepti_size):
    matrix_data = np.array(pd.read_csv(RRC_matrix_csv))
    matrix_Y = matrix_data[0:, 0]
    matrix_X = matrix_data[0:, 1:]

    sample_sum=len(matrix_Y)

    set_X=[]
    for samp_idx in range(0, sample_sum):
        sample_X=[]
        for site in range(0, pepti_size*2+1):
            site_X = matrix_X[samp_idx, site*20:(site+1)*20]
            sample_X.append(site_X)

        set_X.append(sample_X)

    set_X=np.array(set_X)
    return set_X, matrix_Y

def get_RRC_matrix_samp_csv(posi_RRC_matrix_csv, nega_RRC_matrix_csv, pepti_size):
    posi_matrix_X, posi_matrix_Y=get_RRC_matrix_csv(posi_RRC_matrix_csv, pepti_size)
    nega_matrix_X, nega_matrix_Y=get_RRC_matrix_csv(nega_RRC_matrix_csv, pepti_size)

    X = np.concatenate([posi_matrix_X, nega_matrix_X], 0)
    Y = np.concatenate([posi_matrix_Y, nega_matrix_Y], 0)

    return X, Y

def update_auc_arry(add_flag, train_auc, test_auc, train_auc_arry, test_auc_arry, cnt):
    if add_flag == 1:
        train_auc_arry.append(float(train_auc))
        test_auc_arry.append(float(test_auc))
        cnt += 1

    return train_auc_arry, test_auc_arry, cnt

def write_ROC(mean_tpr, mean_fpr, cnt, data_path, model_name):

    mean_tpr /= cnt  # 求数组的平均值
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
    mean_auc = auc(mean_fpr, mean_tpr)
    np.savetxt(data_path + "/ROC_"+model_name+"_mean_fpr.csv", mean_fpr, fmt='%f', delimiter=',')
    np.savetxt(data_path + "/ROC_"+model_name+"_mean_tpr.csv", mean_tpr, fmt='%f', delimiter=',')

    return mean_auc

def get_position_x(position_size, train_count, test_count):
    position_vector = []
    for site in range(0, position_size):
        position_vector.append(site)

    position_train = []
    for index in range(0, train_count):
        position_train.append(position_vector)

    position_test = []
    for index in range(0, test_count):
        position_test.append(position_vector)

    position_train = np.array(position_train)
    position_test = np.array(position_test)

    return position_train,position_test

def EBFF_Kla_GRU(layer_name, model_h5, acid_train_X, contmap_train_X, train_Y, acid_test_X,
                   contmap_test_X, test_Y,
                     epoch, batch_size, mean_tpr, mean_fpr,
                     pepti_size, max_features, Embedding_size,
                     Attension_size, LSTM_unit, dense_size):

    acidNum_input = Input(shape=(pepti_size,))
    acidNum_embed = Embedding(max_features, Embedding_size, name=layer_name+"_acid_embed")(acidNum_input)

    contmap_input = Input(shape=(pepti_size,))
    contmap_embed = Embedding(max_features, Embedding_size, name=layer_name + "_contmap_embed")(contmap_input)

    #merge_inputs = concatenate([acidNum_embed, contmap_embed])
    merge_inputs = Add()([acidNum_embed, contmap_embed])

    acidNum_att = Self_Attention(Attension_size)(merge_inputs)
    acidNum_att = Dropout(0.3)(acidNum_att)
    
    conv_x = Conv1D(filters=LSTM_unit/2, kernel_size=1, name=layer_name+"_conv1", activation='relu')(acidNum_att)
    conv_x = Dropout(0.3)(conv_x)
    
    #acidNum_att_lstm = LSTM(LSTM_unit, return_sequences=False, name=layer_name+"_lstm", kernel_regularizer=l2(0.001),
    #                       bias_regularizer=regularizers.l2(0.0001))(conv_x)

    acidNum_att_bigru =Bidirectional(GRU(32, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.0001),
                       dropout=0.2, return_sequences=False, name=layer_name+"_gru"))(conv_x)

    acidNum_att_lstm = Dropout(0.2)(acidNum_att_bigru)

    merge_dense2 = Dense(dense_size, activation='relu', name=layer_name+"_dense1", kernel_regularizer=l2(1e-5),
                         bias_regularizer=regularizers.l2(0.0001))(acidNum_att_lstm)
    merge_dense2 = Dropout(0.3)(merge_dense2)

    predictions = Dense(1, name=layer_name+"_dense2", activation='sigmoid')(merge_dense2)
    #model = Model(inputs=[acidNum_input, RRC_input], outputs=predictions)
    model = Model(inputs=[acidNum_input, contmap_input], outputs=predictions)

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc', precision])
    model.fit([acid_train_X, contmap_train_X], train_Y, epochs=epoch, batch_size=batch_size,
              validation_data=([acid_test_X, contmap_test_X], test_Y), shuffle=True, verbose=2)

    pre_test_y = model.predict([acid_test_X, contmap_test_X], batch_size=batch_size)
    pre_train_y = model.predict([acid_train_X, contmap_train_X], batch_size=batch_size)

    test_auc = metrics.roc_auc_score(test_Y, pre_test_y)
    train_auc = metrics.roc_auc_score(train_Y, pre_train_y)
    model.save(model_h5 + ".h5")
    print(layer_name+"_train_auc: ", train_auc)
    print(layer_name+"_test_auc: ", test_auc)

    y_pret = pre_test_y.ravel()

    return model, train_auc, test_auc, y_pret, mean_tpr, mean_fpr, add_flag

# split data and output result
##########################################################################
HPC_data_path="/root/kla_contmap/contmap_17/acid_contmap_AddGRU/"
data_path=HPC_data_path

pepti_size=35
contmap_size=35
acid_max_num=22

lr = 0.007 #learning rate
epoch = 80
batch_size = 16
#max_features2=n_bins
#Embedding_size2, conv_size1, Attension_size, conv_size2, biLSTM_units, LSTM_unit

net_param=[64, 32, 32, 32, 16]

Embedding_size = net_param[0]
Attension_size = net_param[1]
LSTM_unit = net_param[2]
knn_LSTM_unit = net_param[3]
dense_size = net_param[4]

print("\nstart net_param=" + str(net_param) + "\n")

kf = KFold(n_splits=10,
           shuffle=True, random_state=57)
kf = kf.split(acidNum_X)

for i, (train_fold, validate_fold) in enumerate(kf):
    print("\n\ni: ", i)

    model1_name="acid_contmap"
    acid_h5_name = data_path + model1_name + str(i)
    acid_model, acid_train_auc, acid_test_auc, acid_y_pret, acid_mean_tpr, acid_mean_fpr, acid_add_flag \
        = EBFF_Kla_GRU(model1_name, acid_h5_name, acidNum_X[train_fold], contmapNum_X[train_fold], Y[train_fold],
                                            acidNum_X[validate_fold], contmapNum_X[validate_fold],Y[validate_fold],
                                                      epoch, batch_size, acid_mean_tpr, acid_mean_fpr,
                                                      pepti_size, acid_max_num, Embedding_size,
                                                      Attension_size, LSTM_unit, dense_size)
