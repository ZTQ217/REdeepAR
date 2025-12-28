import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv(r'D:\final paper\data\^NYA.csv')
# 检查并处理NaN值
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
# 选择特征
features = df[['Open', 'High', 'Low']].values

target = df['Close'].values
# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.reshape(-1, 1))
# 数据切分
features_train, features_test, target_train, target_test = train_test_split(features_scaled, target_scaled, test_size=0.2,
shuffle=False)

# 调整数据形状以符合LSTM输入要求
features_train = np.reshape(features_train, (features_train.shape[0], 1,
features_train.shape[1]))
features_test = np.reshape(features_test, (features_test.shape[0], 1,
features_test.shape[1]))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, 3)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 设置模型权重保存点
checkpoint_filepath = './lstm_checkpoint_best.weights.h5'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
save_weights_only=True,
monitor='val_loss',
mode='min',
save_best_only=True)
# 训练模型
model.fit(features_train, target_train, epochs=100, batch_size=32,
validation_data=(features_test, target_test), callbacks=
[model_checkpoint_callback])
# 加载测试集损失最低的模型权重
model.load_weights(checkpoint_filepath)
# 使用最佳模型权重进行预测
predictions = model.predict(features_test)
# 反归一化预测和真实数据
predictions_inverse = scaler.inverse_transform(predictions)
target_test_inverse = scaler.inverse_transform(target_test)

# 绘制预测值和实际值
plt.figure(figsize=(10, 6))
plt.plot(target_test_inverse, color='blue', label='Actual Close Price')
plt.plot(predictions_inverse, color='red', alpha=0.7, label='Predicted Close Price')
plt.title('RNN-Based Forecasting of NYA Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
# 计算RMSE和MAE
test_rmse = np.sqrt(mean_squared_error(target_test, predictions))
test_mae = mean_absolute_error(target_test, predictions)
print(f'Test RMSE: {test_rmse}')
print(f'Test MAE: {test_mae}')