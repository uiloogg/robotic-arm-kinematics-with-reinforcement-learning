import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from keras.models import Sequential
from keras.layers import Dense

# 設定環境變數與顯示選項
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 5)

# 1. 讀取 Excel 資料
df_THETA = pd.read_excel('AIProj1000.xlsx', sheet_name='THETA')
df_P = pd.read_excel('AIProj1000.xlsx', sheet_name='P')
df_ss_deg = pd.concat([df_THETA, df_P], axis=1)
print(df_ss_deg)

# 2. 資料分區
x_columns = df_ss_deg.columns[1:4]       # x 的欄位範圍
y_product_columns = df_ss_deg.columns[5:8] # y_product 的欄位範圍

# 3. 初始化 MinMaxScaler
ss_x = MinMaxScaler()
ss_y = MinMaxScaler()

# 分配資料
x_train_deg = df_ss_deg[x_columns]
y_train_deg = df_ss_deg[y_product_columns]

# 分割訓練集與測試集
x_train, x_test, y_train, y_test = train_test_split(x_train_deg, y_train_deg, test_size=0.2, random_state=42)

# 正規化資料
x_train_scaled_deg = ss_x.fit_transform(x_train)
y_train_scaled_deg = ss_y.fit_transform(y_train)

# 轉換為 NumPy 陣列
npx_train_scaled_deg = np.array(x_train_scaled_deg)
npy_train_scaled_deg = np.array(y_train_scaled_deg)

# 4. 定義評估指標函式
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r2(actual, predicted):
    ss_Residal = 0.0
    ss_Total = 0.0
    average = np.mean(actual)
    for i in range(len(actual)):
        ss_Residal += (actual[i] - predicted[i]) ** 2
        ss_Total += (actual[i] - average) ** 2
    return 1 - (ss_Residal / ss_Total)

def mse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += prediction_error ** 2
    return sum_error / float(len(actual))

# 5. 建立深度神經網路模型
model = Sequential()
model.add(Dense(units=256, input_dim=npx_train_scaled_deg.shape[1], activation='relu', kernel_initializer='normal'))
model.add(Dense(units=128, activation='tanh', kernel_initializer='normal'))
model.add(Dense(units=64, activation='relu', kernel_initializer='normal'))
model.add(Dense(units=32, activation='tanh', kernel_initializer='normal'))
model.add(Dense(units=3, activation='sigmoid'))

# 6. 編譯與訓練
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.fit(npx_train_scaled_deg, npy_train_scaled_deg, epochs=500, batch_size=32)

# 7. 預測
y_testpred_scaled_deg = model.predict(ss_x.transform(x_test))
y_trainpred_scaled_deg = model.predict(ss_x.transform(x_train))

# 逆轉換 (Inverse Transform)
re_y_testpred_deg = ss_y.inverse_transform(y_testpred_scaled_deg)
re_y_trainpred_deg = ss_y.inverse_transform(y_trainpred_scaled_deg)

# 轉換為陣列進行評估
a_nn = np.array(y_test)
b_nn = np.array(re_y_testpred_deg)
c_nn = np.array(y_train)
d_nn = np.array(re_y_trainpred_deg)

# 8. 輸出結果
print('theta' + '\t\t' + 'R-squared' + '\t\t' + 'RMSE' + '\t\t\t' + 'MAPE' + '\n')
for i in range(len(y_product_columns)):
    if i >= a_nn.shape[1]:
        break
    MSE = mse_metric(a_nn[:, i], b_nn[:, i])
    RMSE = np.sqrt(MSE)
    R_Squared = r2(a_nn[:, i], b_nn[:, i])
    MAPE = mape(a_nn[:, i], b_nn[:, i])
    print(f"{y_product_columns[i]}\t\t{round(R_Squared, 6)}\t\t{round(RMSE, 6)}\t\t{round(MAPE, 6)}")