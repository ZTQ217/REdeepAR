import numpy as np
import pandas as pd
import transformer
from linearmodels.panel import RandomEffects
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import make_evaluation_predictions
from GRU import GRU_forecast
from LSTM import LSTM_forecast
from transformer import transformer_forecast


def normalized_deviation(y_true, y_pred):
    """
    计算归一化偏差 (ND)

    参数:
    y_true -- 实际值 (numpy array 或 list)
    y_pred -- 预测值 (numpy array 或 list)

    返回:
    nd -- 归一化偏差
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        raise ValueError("实际值的总和为0，无法计算ND。")

    nd = numerator / denominator
    return nd


def theils_u(y_true, y_pred):
    """
    计算 Theil's U 统计量

    参数:
    y_true (array-like): 实际值
    y_pred (array-like): 预测值

    返回:
    float: Theil's U 统计量
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 计算分子：RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # 计算分母：实际值和预测值的 RMSE 之和
    rmse_true = np.sqrt(np.mean(y_true ** 2))
    rmse_pred = np.sqrt(np.mean(y_pred ** 2))

    # 计算 Theil's U
    theil_u = rmse / (rmse_true + rmse_pred)

    return theil_u

def quantile_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return (np.maximum(q * error, (q - 1) * error)).mean()


data = pd.read_csv('000001.SZall.csv')

data['id'] = '000001.SZall.csv'
train_size = int(data.shape[0] * 0.8)
test_size = data.shape[0] - train_size

panel_data = pd.concat([data], ignore_index=True)

panel_data['trade_date'] = pd.to_datetime(panel_data['trade_date'], format='%Y%m%d')
panel_data.set_index(['id', 'trade_date'], inplace=True)
train_data_RE = panel_data[:train_size]
test_data_RE = panel_data[train_size:]

dependent_var = panel_data['close']
independent_vars = panel_data[['pb', 'ps', 'pe', 'turnover_rate', 'dv_ratio']]

independent_vars = sm.add_constant(independent_vars)
y_RE_train = dependent_var[:train_size]
X_RE_train = independent_vars[:train_size]
y_RE_test = dependent_var[train_size:]
X_RE_test = independent_vars[train_size:]


re_model = RandomEffects(y_RE_train, X_RE_train)
re_results = re_model.fit()

test_data_RE = test_data_RE.dropna()
test_data_RE.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data_RE.dropna(inplace=True)
RE_predictions = re_results.predict(exog=X_RE_test)
test_data_RE['predicted'] = RE_predictions
df_plot = test_data_RE[['close', 'predicted']].unstack(level='id')

for stock_id in df_plot['close'].columns:
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot['close'][stock_id], label='Actual')
    plt.plot(df_plot['predicted'][stock_id], label='Predicted', alpha=0.7)
    plt.title(f'RandomEffect model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig("./RandomEffectrealdataprediction.jpg")
    plt.show()
    plt.close()

r2_RE = r2_score(test_data_RE['close']['000001.SZall.csv'], test_data_RE['predicted']['000001.SZall.csv'])
rmse_RE = np.sqrt(mean_squared_error(test_data_RE['close']['000001.SZall.csv'], test_data_RE['predicted']['000001.SZall.csv']))
cos_RE = np.dot(test_data_RE['close']['000001.SZall.csv'], test_data_RE['predicted']['000001.SZall.csv']) / (np.linalg.norm(test_data_RE['close']['000001.SZall.csv']) * np.linalg.norm(test_data_RE['predicted']['000001.SZall.csv']))
ND_RE = normalized_deviation(test_data_RE['close']['000001.SZall.csv'], test_data_RE['predicted']['000001.SZall.csv'])
U_RE = theils_u(test_data_RE['close']['000001.SZall.csv'], test_data_RE['predicted']['000001.SZall.csv'])
print(f'R^2_RE: {r2_RE}')
print(f'RMSE_RE:{rmse_RE}')

data = pd.read_csv('000001.SZall.csv')
#data['id'] = '000001.SZ'
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
#data.set_index(['id', 'trade_date'], inplace=True)


data_train = data[:train_size]
train_data = data[:train_size]
test_data = data[train_size:]

GRU_test = GRU_forecast(data)
LSTM_test = LSTM_forecast(data)
transformer_test = transformer_forecast(data)

r2_GRU = r2_score(test_data['close'], np.array(GRU_test).flatten())
rmse_GRU = np.sqrt(mean_squared_error(test_data['close'], np.array(GRU_test).flatten()))
cos_GRU = np.dot(test_data['close'], GRU_test) / (np.linalg.norm(test_data['close']) * np.linalg.norm(np.array(GRU_test).flatten()))
ND_GRU = normalized_deviation(test_data['close'], np.array(GRU_test).flatten())
U_GRU = theils_u(test_data['close'], np.array(GRU_test).flatten())
r2_LSTM = r2_score(test_data['close'], np.array(LSTM_test).flatten())
rmse_LSTM = np.sqrt(mean_squared_error(test_data['close'], np.array(LSTM_test).flatten()))
cos_LSTM = np.dot(test_data['close'], GRU_test) / (np.linalg.norm(test_data['close']) * np.linalg.norm(np.array(LSTM_test).flatten()))
ND_LSTM = normalized_deviation(test_data['close'], np.array(LSTM_test).flatten())
U_LSTM = theils_u(test_data['close'], np.array(LSTM_test).flatten())
r2_transf = r2_score(test_data['close'], np.array(transformer_test).flatten())
rmse_transf = np.sqrt(mean_squared_error(test_data['close'], np.array(transformer_test).flatten()))
cos_transf = np.dot(test_data['close'], GRU_test) / (np.linalg.norm(test_data['close']) * np.linalg.norm(np.array(transformer_test).flatten()))
ND_transf = normalized_deviation(test_data['close'], np.array(transformer_test).flatten())
U_transf = theils_u(test_data['close'], np.array(transformer_test).flatten())

training_data = ListDataset(
    [{'start':  train_data['trade_date'].iloc[0],
      'target': train_data['close'].to_numpy()}],
    freq='D'
)

testing_data = ListDataset(
    [{'start': test_data['trade_date'].iloc[0],
      'target': test_data['close'].to_numpy()}],
    freq='D'
)

estimator = DeepAREstimator(
    prediction_length=1,
    context_length=36,
    freq="D",
    trainer=Trainer(epochs=1)
)
predictor = estimator.train(training_data=training_data)


forecasts = []
for i in range(test_size):
    single_step_training_data = ListDataset(
        [{"start": test_data['trade_date'].iloc[0], "target": test_data['close'][:i + 1].values}],
        freq="D"
    )

    forecast_entry = next(predictor.predict(single_step_training_data))

    forecasts.append(forecast_entry.mean[0])


r2_deepAR = r2_score(test_data['close'], forecasts)
rmse_deepAR = np.sqrt(mean_squared_error(test_data['close'], forecasts))
cos_deepAR = np.dot(test_data['close'], forecasts) / (np.linalg.norm(test_data['close']) * np.linalg.norm(forecasts))
ND_deepAR = normalized_deviation(test_data['close'], np.array(forecasts).flatten())
U_deepAR = theils_u(test_data['close'], np.array(forecasts).flatten())

plt.figure(figsize=(12, 6))
plt.plot(test_data['trade_date'], test_data['close'], label='Actual', linewidth=1)
plt.plot(test_data['trade_date'], forecasts, label='Forecast', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('DeepAR model')
plt.legend()
plt.savefig("./DeepARrealdataprediction.jpg")
plt.show()
plt.close()

print(f'R^2_deepAR: {r2_deepAR}')
print(f'RMSE_deepAR:{rmse_deepAR}')

model = ARIMA(train_data['close'].to_numpy(), order=(1, 1, 1))
model_fit = model.fit()

# 预测
y_ARIMA = model_fit.forecast(steps=test_size)
# Store the model predicted y values for the test set

r2_ARIMA = r2_score(test_data['close'], y_ARIMA)
rmse_ARIMA = np.sqrt(mean_squared_error(test_data['close'], y_ARIMA))
cos_ARIMA = np.dot(test_data['close'], y_ARIMA) / (np.linalg.norm(test_data['close']) * np.linalg.norm(y_ARIMA))
ND_ARIMA = normalized_deviation(test_data['close'], np.array(y_ARIMA).flatten())
U_ARIMA = theils_u(test_data['close'], np.array(y_ARIMA).flatten())

data = pd.read_csv('000001.SZall.csv')
data['id'] = '000001.SZ'
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
date = data['trade_date']
data.set_index(['id', 'trade_date'], inplace=True)

y = data['close']
y_train = y[:train_size]
X = data.drop(columns='close')

X_const = sm.add_constant(X)
X_const_train = X_const[:train_size]
X_const_test = X_const[train_size:]
re_model = RandomEffects(y_train, X_const_train)
re_results = re_model.fit()
data['predicted_y'] = re_results.predict(X_const)
data['residuals'] = y - data['predicted_y']
data['trade_date'] = date

train_data = data[:train_size]
test_data = data[train_size:]

training_res = ListDataset(
    [{'start': train_data.index.get_level_values('trade_date')[0],
      'target': train_data['residuals'].to_numpy()
      }],
    freq='D'
)

testing_res = ListDataset(
    [{'start': test_data.index.get_level_values('trade_date')[0],
      'target': test_data['residuals'].to_numpy()
    }],
    freq='D'
)
"""
estimator = DeepAREstimator(
    freq='D',
    prediction_length=1,
    context_length=14,
    num_layers=2,
    num_cells=40,
    trainer=Trainer(epochs=30)
)
"""
predictor = estimator.train(training_data=training_res)


RE_forecasts = []
for i in range(test_size):
    single_step_training_data = ListDataset(
        [{"start": test_data.index.get_level_values('trade_date')[0], "target": test_data['residuals'][:i + 1].values}],
        freq="D"
    )

    forecast_entry = next(predictor.predict(single_step_training_data))

    RE_forecasts.append(forecast_entry.mean[0])

re_prediction = re_results.predict(X_const.loc[test_data.index])
if len(re_prediction.shape) > 1:
    re_prediction = re_prediction.squeeze()

test_data_forecast = re_prediction.values + np.array(RE_forecasts)
#test_data.dropna(subset=['final_predicted_y'], inplace=True)

r2_res = r2_score(test_data['close'], test_data_forecast)
rmse_res = np.sqrt(mean_squared_error(test_data['close'], test_data_forecast))
cos_res = np.dot(test_data['close'], test_data_forecast) / (np.linalg.norm(test_data['close']) * np.linalg.norm(test_data_forecast))
ND_res = normalized_deviation(test_data['close'], np.array(test_data_forecast).flatten())
U_res = theils_u(test_data['close'], np.array(test_data_forecast).flatten())

plt.figure(figsize=(12, 6))
plt.plot(test_data.index.get_level_values('trade_date'), test_data['close'], label='Actual', linestyle='--', color = 'blue')
plt.plot(test_data.index.get_level_values('trade_date'), test_data_forecast, label='REdeepAR',
         linestyle='--', color = 'red')
plt.plot(test_data.index.get_level_values('trade_date'), forecasts, label='deepAR', linestyle='--', color = 'green')
plt.plot(test_data.index.get_level_values('trade_date'), test_data_RE['predicted'], label='RandomEffect', linestyle='--', color = 'grey')
plt.plot(test_data.index.get_level_values('trade_date'), y_ARIMA, label='ARIMA', linestyle='--', color = 'yellow')
plt.plot(test_data.index.get_level_values('trade_date'), GRU_test, label='GRU', linestyle='-', color = 'm')
plt.plot(test_data.index.get_level_values('trade_date'), LSTM_test, label='LSTM', linestyle='-', color = 'pink')
plt.plot(test_data.index.get_level_values('trade_date'), transformer_test, label='transformer', linestyle='--', color = 'blueviolet')
plt.title('Four Methods Compare')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig("./Threemethodsrealdataprediction.jpg")
plt.show()
plt.close()

print(f'R^2: {r2_res}')
print(f'RMSE:{rmse_res}')
est_mean = np.zeros(35).reshape(7, 5)
est_mean[0, 0] = r2_RE
est_mean[1, 0] = r2_deepAR
est_mean[2, 0] = r2_ARIMA
est_mean[3, 0] = r2_res
est_mean[4, 0] = r2_GRU
est_mean[5, 0] = r2_LSTM
est_mean[6, 0] = r2_transf

est_mean[0, 1] = rmse_RE
est_mean[1, 1] = rmse_deepAR
est_mean[2, 1] = rmse_ARIMA
est_mean[3, 1] = rmse_res
est_mean[4, 1] = rmse_GRU
est_mean[5, 1] = rmse_LSTM
est_mean[6, 1] = rmse_transf

est_mean[0, 2] = cos_RE
est_mean[1, 2] = cos_deepAR
est_mean[2, 2] = cos_ARIMA
est_mean[3, 2] = cos_res
est_mean[4, 2] = cos_GRU
est_mean[5, 2] = cos_LSTM
est_mean[6, 2] = cos_transf

est_mean[0, 3] = ND_RE
est_mean[1, 3] = ND_deepAR
est_mean[2, 3] = ND_ARIMA
est_mean[3, 3] = ND_res
est_mean[4, 3] = ND_GRU
est_mean[5, 3] = ND_LSTM
est_mean[6, 3] = ND_transf

est_mean[0, 4] = U_RE
est_mean[1, 4] = U_deepAR
est_mean[2, 4] = U_ARIMA
est_mean[3, 4] = U_res
est_mean[4, 4] = U_GRU
est_mean[5, 4] = U_LSTM
est_mean[6, 4] = U_transf

alpha_result = pd.DataFrame(est_mean, columns=["R2", "RMSE", "COS", "ND", "U"])
alpha_result.index = ["RE", "deepAR", "ARIMA", "REdeepAR", "GRU", "LSTM", "transformer"]
alpha_result.to_csv("./statisticalindicatoractrualdata.csv")

Q_loss = np.zeros(21).reshape(3, 7)
ie = 0
for q in [0.5, 0.75, 0.9]:
    RE_qloss = quantile_loss(test_data['close'], test_data_RE['predicted'].values.flatten(), q)
    DeepAR_qloss = quantile_loss(test_data['close'], np.array(forecasts).flatten(), q)
    ARIMA_qloss = quantile_loss(test_data['close'], np.array(y_ARIMA).flatten(), q)
    GRU_qloss = quantile_loss(test_data['close'], np.array(GRU_test).flatten(), q)
    LSTM_qloss = quantile_loss(test_data['close'], np.array(LSTM_test).flatten(), q)
    transformer_qloss = quantile_loss(test_data['close'], np.array(transformer_test).flatten(), q)
    REdeepAR_qloss = quantile_loss(test_data['close'], np.array(test_data_forecast).flatten(), q)
    Q_loss[ie] = np.array([RE_qloss, DeepAR_qloss, ARIMA_qloss, GRU_qloss, LSTM_qloss, transformer_qloss, REdeepAR_qloss])
    ie = ie + 1
qloss_result = pd.DataFrame(Q_loss, columns=["RE", "deepAR", "ARIMA", "REdeepAR", "GRU", "LSTM", "transformer"])
qloss_result.index = ["0.5", "0.75", "0.9"]
qloss_result.to_csv("./qlossactrualdata.csv")