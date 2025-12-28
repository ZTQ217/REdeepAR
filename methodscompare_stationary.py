import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import RandomEffects
#from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from MCS import MCS

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


def moving_average(data, window_size=10):
    """Compute the moving average of a given data using a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Set parameters
n = 2500  # sample size
p = 4    # number of parameters
num_simulations = 100  # number of simulations
phi = 0.8   # 自回归参数 (p=2) 
d = 1                # 差分次数
train_size = 2000
test_size = n - train_size
entity_ids = np.repeat(np.arange(1), n)
time_ids = np.tile(np.arange(n), 1)


np.random.seed(1)
true_beta = np.random.uniform(-0.5, 0.5, 4)

# 生成具有自相关性的随机效应
# 生成具有自相关性的随机效应
random_effects = np.zeros(n)
for i in range(1, n):
    np.random.seed(n+p+i)
    random_effects[i] = phi * random_effects[i-1] + np.random.normal(0, 1, 1)

# 用于存储预测结果的列表
y_real = []
y_REdeepAR = []
y_deepARestimates = []
y_REestimates = []
y_ARIMAestimates = []
R2_RE = []
R2_deepAR = []
R2_REdeepAR = []
R2_ARIMA = []
RMSE_RE = []
RMSE_deepAR = []
RMSE_REdeepAR = []
RMSE_ARIMA = []
COS_RE =[]
COS_deepAR=[]
COS_REdeepAR = []
COS_ARIMA = []
U_RE =[]
U_deepAR=[]
U_REdeepAR = []
U_ARIMA = []
ND_RE =[]
ND_deepAR=[]
ND_REdeepAR = []
ND_ARIMA = []
MSE_models = np.zeros(4)
MSEP_values = np.zeros((num_simulations, 4))
MAE_models = np.zeros(4)
MAEP_values = np.zeros((num_simulations, 4))
qloss5_models = np.zeros(4)
qloss5_values = np.zeros((num_simulations, 4))
qloss9_models = np.zeros(4)
qloss9_values = np.zeros((num_simulations, 4))

# 执行模拟
for j in range(num_simulations):
    # 添加噪声以生成观测值
    # 计算真实的y值
    np.random.seed(n + p + j)
    # 生成X值
    X = np.random.randn(n, p)
    epsilon = np.random.normal(0, 0.5, n)
    y_true = X.dot(true_beta) + random_effects + epsilon

    index = pd.MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])
    df = pd.DataFrame(X, index=index, columns=[f"x{i}" for i in range(p)])
    df['y'] = y_true
    y = df['y']
    X = df.drop(columns='y')
    X = sm.add_constant(X)

    # 数据集划分
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    y_real.append(y_test)

    model = RandomEffects(y_train, X_train)
    result = model.fit()
    y_REfit = result.predict(exog=X)

    model = ARIMA(y_train.to_numpy(), order=(1,1,1))
    model_fit = model.fit()

    # 预测
    y_ARIMA = model_fit.forecast(steps=test_size)
    # Store the model predicted y values for the test set

    y_ARIMAestimates.append(y_ARIMA)
    r2_ARIMA = r2_score(y_test, y_ARIMA)
    R2_ARIMA.append(r2_ARIMA)
    rmse_ARIMA = np.sqrt(mean_squared_error(y_test, y_ARIMA))
    RMSE_ARIMA.append(rmse_ARIMA)
    cos_ARIMA = np.dot(y_test, y_ARIMA) / (np.linalg.norm(y_test) * np.linalg.norm(y_ARIMA))
    COS_ARIMA.append(cos_ARIMA)
    u_arima = theils_u(y_test.values.flatten(), y_ARIMA)
    U_ARIMA.append(u_arima)
    nd_arima = normalized_deviation(y_test.values.flatten(), y_ARIMA)
    ND_ARIMA.append(nd_arima)

    # 计算残差
    residuals = y_true - y_REfit.values.reshape(n, )
    res_train = residuals[:train_size]
    res_test = residuals[train_size:]

    # Store the model predicted y values for the test set
    y_RE = result.predict(exog=X_test)
    y_REestimates.append(y_RE)
    r2_RE = r2_score(y_test, y_RE)
    R2_RE.append(r2_RE)
    rmse_RE = np.sqrt(mean_squared_error(y_test, y_RE))
    RMSE_RE.append(rmse_RE)
    cos_RE = np.dot(y_test, y_RE) / (np.linalg.norm(y_test) * np.linalg.norm(y_RE))
    COS_RE.append(cos_RE)
    u_RE = theils_u(y_test.values.flatten(), y_RE.values.flatten())
    U_RE.append(u_RE)
    nd_RE = normalized_deviation(y_test.values.flatten(), y_RE.values.flatten())
    ND_RE.append(nd_RE)
    # 将训练数据转换为GluonTS所需的格式
    train_data = [{'start': pd.Timestamp("2020-01-01", freq='D'), 'target': y_train}]
    training_data = ListDataset(train_data, freq='D')
    test_data = [{'start': pd.Timestamp("2020-01-01", freq='D'), 'target': y_test}]
    testing_data = ListDataset(test_data, freq='D')
    #test_value = y_test.reshape(test_size,)
    # 定义DeepAR模型
    estimator = DeepAREstimator(
        prediction_length=1,
        context_length=36,
        freq="D",
        trainer=Trainer(epochs=1)
    )
    predictor = estimator.train(training_data=training_data)

    y_forecasts = []
    for i in range(test_size):
        single_step_training_data = ListDataset(
            [{"start": pd.Timestamp("2020-01-01", freq='D'), "target": y_test[:i + 1]}],
            freq="D"
        )

        forecast_entry = next(predictor.predict(single_step_training_data))

        y_forecasts.append(forecast_entry.mean[0])
    r2_deepAR = r2_score(y_test, y_forecasts)
    rmse_deepAR = np.sqrt(mean_squared_error(y_test, y_forecasts))
    cos_deepAR = np.dot(y_test, y_forecasts) / (np.linalg.norm(y_test) * np.linalg.norm(y_forecasts))
    R2_deepAR.append(r2_deepAR)
    RMSE_deepAR.append(rmse_deepAR)
    COS_deepAR.append(cos_deepAR)
    u_deepAR = theils_u(y_test.values.flatten(), np.array(y_forecasts))
    U_deepAR.append(u_deepAR)
    nd_deepAR = normalized_deviation(y_test.values.flatten(), np.array(y_forecasts))
    ND_deepAR.append(nd_deepAR)
    # y_pred = [forecast.mean[0] for forecast in forecasts]

    # 存储模型的预测y值
    y_deepARestimates.append(y_forecasts)

    train_res = [{'start': pd.Timestamp("2020-01-01", freq='D'), 'target': res_train}]
    training_res = ListDataset(train_data, freq='D')
    test_res = [{'start': pd.Timestamp("2020-01-01", freq='D'), 'target': res_test}]
    testing_res = ListDataset(test_data, freq='D')

    predictor = estimator.train(training_data=training_res)

    res_forecasts = []
    for i in range(test_size):
        single_step_training_data = ListDataset(
            [{"start": pd.Timestamp("2020-01-01", freq='D'), "target": res_test[:i + 1]}],
            freq="D"
        )

        forecast_entry = next(predictor.predict(single_step_training_data))

        res_forecasts.append(forecast_entry.mean[0])
    y_REforecasts = y_RE.values.reshape(test_size, ) + res_forecasts
    y_REdeepAR.append(y_REforecasts)
    r2_res = r2_score(y_test, y_REforecasts)
    rmse_res = np.sqrt(mean_squared_error(y_test, y_REforecasts))
    R2_REdeepAR.append(r2_res)
    RMSE_REdeepAR.append(rmse_res)
    cos_REdeepAR = np.dot(y_test, y_REforecasts) / (np.linalg.norm(y_test) * np.linalg.norm(y_REforecasts))
    COS_REdeepAR.append(cos_REdeepAR)
    u_REdeepAR = theils_u(y_test.values.flatten(), np.array(y_REforecasts))
    U_REdeepAR.append(u_REdeepAR)
    nd_REdeepAR = normalized_deviation(y_test.values.flatten(), np.array(y_REforecasts))
    ND_REdeepAR.append(nd_REdeepAR)

    err_res = y_test.values.flatten() - np.array(y_REforecasts)
    MSE_res = err_res ** 2
    MAE_res = np.abs(err_res)
    qloss5_res = np.maximum(0.5 * err_res, (0.5 - 1) * err_res)
    qloss9_res = np.maximum(0.9 * err_res, (0.9 - 1) * err_res)

    err_deepAR = y_test.values.flatten() - np.array(y_forecasts)
    MSE_deepAR = err_deepAR ** 2
    MAE_deepAR = np.abs(err_deepAR)
    qloss5_deepAR = np.maximum(0.5 * err_deepAR, (0.5 - 1) * err_deepAR)
    qloss9_deepAR = np.maximum(0.9 * err_deepAR, (0.9 - 1) * err_deepAR)

    err_RE = y_test.values.flatten() - y_RE.values.flatten()
    MSE_RE = err_RE ** 2
    MAE_RE = np.abs(err_RE)
    qloss5_RE = np.maximum(0.5 * err_RE, (0.5 - 1) * err_RE)
    qloss9_RE = np.maximum(0.9 * err_RE, (0.9 - 1) * err_RE)

    err_ARIMA = y_test.values.flatten() - y_ARIMA
    MSE_ARIMA = err_ARIMA ** 2
    MAE_ARIMA = np.abs(err_ARIMA)
    qloss5_ARIMA = np.maximum(0.5 * err_ARIMA, (0.5 - 1) * err_ARIMA)
    qloss9_ARIMA = np.maximum(0.9 * err_ARIMA, (0.9 - 1) * err_ARIMA)

    MSE = np.array([MSE_res, MSE_deepAR, MSE_ARIMA, MSE_RE])
    MSE_mcs = MCS(MSE)
    MSE_models += MSE_mcs["modelindex"]
    MSEP_values[j] = MSE_mcs["P_values"]
    MAE = np.array([MAE_res, MAE_deepAR, MAE_ARIMA, MAE_RE])
    MAE_mcs = MCS(MAE)
    MAE_models += MAE_mcs["modelindex"]
    MAEP_values[j] = MAE_mcs["P_values"]
    qloss5 = np.array([qloss5_res, qloss5_deepAR, qloss5_ARIMA, qloss5_RE])
    qloss5_mcs = MCS(qloss5)
    qloss5_models += qloss5_mcs["modelindex"]
    qloss5_values[j] = qloss5_mcs["P_values"]
    qloss9 = np.array([qloss9_res, qloss9_deepAR, qloss9_ARIMA, qloss9_RE])
    qloss9_mcs = MCS(qloss9)
    qloss9_models += qloss9_mcs["modelindex"]
    qloss9_values[j] = qloss9_mcs["P_values"]

MCS = np.array([MSE_models, MAE_models, qloss5_models, qloss9_models]) / num_simulations
MCS_result = pd.DataFrame(MCS.T, columns=["MSE", "MAE", "qloss_0.5", "qloss_0.9"])
MCS_result.index = ["REdeepAR", "DeepAR", "ARIMA", "RandomEffect"]
MCS_result.to_csv("stationaryAR1MCSpercent.csv")
MSEP_values_mean = np.nanmean(MSEP_values, axis=0)
MAEP_values_mean = np.nanmean(MAEP_values, axis=0)
qloss5_values_mean = np.nanmean(qloss5_values, axis=0)
qloss9_values_mean = np.nanmean(qloss9_values, axis=0)
P_values = np.array([MSEP_values_mean, MAEP_values_mean, qloss5_values_mean, qloss9_values_mean])
Pvalues_result = pd.DataFrame(P_values.T, columns=["MSE", "MAE", "qloss_0.5", "qloss_0.9"])
Pvalues_result.index = ["REdeepAR", "DeepAR", "ARIMA", "RandomEffect"]
Pvalues_result.to_csv("stationaryAR1MCSpvalues.csv")

# 将y估计值转换为numpy数组以进行进一步处理
y_real_array = np.array(y_real)
y_REestimates_array = np.array(y_REestimates)
y_RE_mean = np.mean(y_REestimates_array, axis=0)
y_RE_std = np.std(y_REestimates_array, axis=0)

y_deepARestimates_array = np.array(y_deepARestimates)
y_deepAR_mean = np.mean(y_deepARestimates_array, axis=0)
y_deepAR_std = np.std(y_deepARestimates_array, axis=0)

y_REdeepAR_array = np.array(y_REdeepAR)
y_REdeepAR_mean = np.mean(y_REdeepAR_array, axis=0)
y_REdeepAR_std = np.std(y_REdeepAR_array, axis=0)

y_ARIMA_array = np.array(y_ARIMAestimates)
y_ARIMA_mean = np.mean(y_ARIMA_array, axis=0)
y_ARIMA_std = np.std(y_ARIMA_array, axis=0)

y_real_mean = np.mean(y_real_array, axis=0)
y_real_median = np.median(y_real_array, axis=0)
# For visualization purposes, we only take the moving average of y_true for the test set
y_truemean_smoothed = moving_average(y_real_mean, window_size=5)
y_truemedian_smoothed = moving_average(y_real_median, window_size=5)

# 计算置信区间
y_REupper = y_RE_mean + y_RE_std
y_RElower = y_RE_mean - y_RE_std

# Apply the same window to the upper and lower bounds for consistency
y_REupper_smoothed = moving_average(y_REupper.reshape(test_size,), window_size=5)
y_RElower_smoothed = moving_average(y_RElower.reshape(test_size,), window_size=5)

# 计算估计的y值的50th百分位数（中位数）
y_REmedian = np.percentile(y_REestimates_array, 50, axis=0)

# Smooth the median estimated y values
y_REmedian_smoothed = moving_average(y_REmedian.reshape(test_size,), window_size=5)

# 计算置信区间
y_ARIMAupper = y_ARIMA_mean + y_ARIMA_std
y_ARIMAlower = y_ARIMA_mean - y_ARIMA_std

# Apply the same window to the upper and lower bounds for consistency
y_ARIMAupper_smoothed = moving_average(y_ARIMAupper.reshape(test_size,), window_size=5)
y_ARIMAlower_smoothed = moving_average(y_ARIMAlower.reshape(test_size,), window_size=5)

# 计算估计的y值的50th百分位数（中位数）
y_ARIMAmedian = np.percentile(y_ARIMA_array, 50, axis=0)

# Smooth the median estimated y values
y_ARIMAmedian_smoothed = moving_average(y_ARIMAmedian.reshape(test_size,), window_size=5)


# 计算置信区间
y_deepARupper = y_deepAR_mean + y_deepAR_std
y_deepARlower = y_deepAR_mean - y_deepAR_std

# Apply the same window to the upper and lower bounds for consistency
y_deepARupper_smoothed = moving_average(y_deepARupper, window_size=5)
y_deepARlower_smoothed = moving_average(y_deepARlower, window_size=5)

# 计算估计的y值的50th百分位数（中位数）
y_deepARmedian = np.percentile(y_deepARestimates_array, 50, axis=0)

# Smooth the median estimated y values
y_deepARmedian_smoothed = moving_average(y_deepARmedian, window_size=5)

# 计算置信区间
y_REdeepARupper = y_REdeepAR_mean + y_REdeepAR_std
y_REdeepARlower = y_REdeepAR_mean - y_REdeepAR_std

# Apply the same window to the upper and lower bounds for consistency
y_REdeepARupper_smoothed = moving_average(y_REdeepARupper, window_size=5)
y_REdeepARlower_smoothed = moving_average(y_REdeepARlower, window_size=5)

# 计算估计的y值的50th百分位数（中位数）
y_REdeepARmedian = np.percentile(y_REdeepAR_array, 50, axis=0)

# Smooth the median estimated y values
y_REdeepARmedian_smoothed = moving_average(y_REdeepARmedian, window_size=5)

# 绘制结果
plt.figure(figsize=(15, 6))
plt.fill_between(range(len(y_ARIMAupper_smoothed)), y_ARIMAlower_smoothed, y_ARIMAupper_smoothed, color='grey', alpha=0.5, label='Confidence Interval (3σ)')
plt.plot(y_real_mean, label="True y", color='blue')
plt.plot(y_ARIMA_mean, label="Mean Estimated y", linestyle='--', color='red')
plt.title("ARIMA Model on Test Set")
plt.xlabel("Observation")
plt.ylabel("y Value")
plt.legend()
plt.tight_layout()
plt.savefig("stationaryAR1ARIMApredictsimulation.jpg")
plt.show()
plt.close()

# 绘制结果
plt.figure(figsize=(15, 6))
plt.fill_between(range(len(y_REupper_smoothed)), y_RElower_smoothed, y_REupper_smoothed, color='grey', alpha=0.5, label='Confidence Interval (3σ)')
plt.plot(y_real_mean, label="True y", color='blue')
plt.plot(y_RE_mean, label="Mean Estimated y", linestyle='--', color='red')
plt.title("RandomEffect Model on Test Set")
plt.xlabel("Observation")
plt.ylabel("y Value")
plt.legend()
plt.tight_layout()
plt.savefig("stationaryAR1RandomEffectpredictsimulation.jpg")
plt.show()
plt.close()

plt.figure(figsize=(15, 6))
plt.fill_between(range(len(y_deepARupper_smoothed)), y_deepARlower_smoothed, y_deepARupper_smoothed, color='grey', alpha=0.5, label='Confidence Interval (3σ)')
plt.plot(y_real_mean, label="True y", color='blue')
plt.plot(y_deepAR_mean, label="Mean Estimated y", linestyle='--', color='red')
plt.title("DeepAR Model on Test Set")
plt.xlabel("Observation")
plt.ylabel("y Value")
plt.legend()
plt.tight_layout()
plt.savefig("stationaryAR1DeepARpredictsimulation.jpg")
plt.show()
plt.close()

plt.figure(figsize=(15, 6))
plt.fill_between(range(len(y_REdeepARupper_smoothed)), y_REdeepARlower_smoothed, y_REdeepARupper_smoothed, color='grey', alpha=0.5, label='Confidence Interval (3σ)')
plt.plot(y_real_mean, label="True y", color='blue')
plt.plot(y_REdeepAR_mean, label="Mean Estimated y", linestyle='--', color='red')
plt.title("REDeepAR Model on Test Set")
plt.xlabel("Observation")
plt.ylabel("y Value")
plt.legend()
plt.tight_layout()
plt.savefig("stationaryAR1RandomEffectDeepARpredictsimulation.jpg")
plt.show()
plt.close()

plt.figure(figsize=(15, 6))
plt.plot(y_real_mean, label="True y", linestyle='--', color='blue')
plt.plot(y_REdeepAR_mean, label="REdeepAR Estimated y", color='red')
plt.plot(y_deepAR_mean, label="deepAR Estimated y", color='green')
plt.plot(y_RE_mean, label="RE Estimated y", color='grey')
plt.plot(y_ARIMA_mean, label="ARIMA Estimated y", color='yellow')
plt.title("compare three methods")
plt.xlabel("Observation")
plt.ylabel("y Value")
plt.legend()
plt.tight_layout()
plt.savefig("stationaryAR1Comparethreemethodspredictsimulation.jpg")
plt.show()
plt.close()

R2mean_RE = np.mean(np.array(R2_RE))
R2mean_deepAR = np.mean(np.array(R2_deepAR))
R2mean_REdeepAR = np.mean(np.array(R2_REdeepAR))
R2mean_ARIMA = np.mean(np.array(R2_ARIMA))

RMSEmean_RE = np.mean(np.array(RMSE_RE))
RMSEmean_deepAR = np.mean(np.array(RMSE_deepAR))
RMSEmean_REdeepAR = np.mean(np.array(RMSE_REdeepAR))
RMSEmean_ARIMA = np.mean(np.array(RMSE_ARIMA))

COSmean_RE = np.mean(np.array(COS_RE))
COSmean_deepAR = np.mean(np.array(COS_deepAR))
COSmean_REdeepAR = np.mean(np.array(COS_REdeepAR))
COSmean_ARIMA = np.mean(np.array(COS_ARIMA))
Umean_RE = np.mean(np.array(U_RE))
Umean_deepAR = np.mean(np.array(U_deepAR))
Umean_REdeepAR = np.mean(np.array(U_REdeepAR))
Umean_ARIMA = np.mean(np.array(U_ARIMA))

NDmean_RE = np.mean(np.array(ND_RE))
NDmean_deepAR = np.mean(np.array(ND_deepAR))
NDmean_REdeepAR = np.mean(np.array(ND_REdeepAR))
NDmean_ARIMA = np.mean(np.array(ND_ARIMA))

est_mean = np.zeros(20).reshape(4, 5)
est_mean[0, 0] = np.mean(R2mean_RE)
est_mean[1, 0] = np.mean(R2mean_deepAR)
est_mean[2, 0] = np.mean(R2mean_ARIMA)
est_mean[3, 0] = np.mean(R2mean_REdeepAR)

est_mean[0, 1] = np.mean(RMSE_RE)
est_mean[1, 1] = np.mean(RMSE_deepAR)
est_mean[2, 1] = np.mean(RMSE_ARIMA)
est_mean[3, 1] = np.mean(RMSE_REdeepAR)

est_mean[0, 2] = np.mean(COS_RE)
est_mean[1, 2] = np.mean(COS_deepAR)
est_mean[2, 2] = np.mean(COS_ARIMA)
est_mean[3, 2] = np.mean(COS_REdeepAR)

est_mean[0, 3] = np.mean(U_RE)
est_mean[1, 3] = np.mean(U_deepAR)
est_mean[2, 3] = np.mean(U_ARIMA)
est_mean[3, 3] = np.mean(U_REdeepAR)

est_mean[0, 4] = np.mean(ND_RE)
est_mean[1, 4] = np.mean(ND_deepAR)
est_mean[2, 4] = np.mean(ND_ARIMA)
est_mean[3, 4] = np.mean(ND_REdeepAR)

alpha_result = pd.DataFrame(est_mean, columns=["R2", "RMSE", "COS", "U", "ND"])
alpha_result.index = ["RE", "deepAR", "ARIMA", "REdeepAR"]
alpha_result.to_csv("./stationaryAR1predictioninstructuremean.csv")
