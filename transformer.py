import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras import Input, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Transformer Encoder Layer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
    key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

# Build the model
def build_model(input_shape, head_size, num_heads, ff_dim,
   num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
       x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for units in mlp_units:
        x = Dense(units, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

# Data Preparation
df = pd.read_csv('D:/桌面2/是/论文/毕业论文/^NYA.csv').fillna(method='ffill').fillna(method='bfill')
features = df[['Open', 'High', 'Low']].values
target = df['Close'].values
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.reshape(-1, 1))
features_train, features_test, target_train, target_test = train_test_split(
    features_scaled, target_scaled, test_size=0.2,
shuffle=False)
features_train = np.expand_dims(features_train, axis=1)
features_test = np.expand_dims(features_test, axis=1)

# Model Configuration
model = build_model(
    input_shape=features_train.shape[1:],
    head_size=64,
    num_heads=2,
    ff_dim=32,
    num_transformer_blocks=1,
    mlp_units=[64],
    dropout=0.1,
    mlp_dropout=0.1
)

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()
# Train the model
model.fit(features_train, target_train, epochs=200, batch_size=32,
verbose=1)
# Evaluate the model
predictions = model.predict(features_test)
# Inverse Transform to get actual values
inverse_predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(target_test)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actual, label='Actual Close Price')
plt.plot(inverse_predictions, label='Predicted Close Price', alpha=0.7)
plt.title('transformer-Based Forecasting of NYA Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
test_rmse = np.sqrt(mean_squared_error(target_test, predictions))
test_mae = mean_absolute_error(target_test, predictions)
print(f'Test RMSE: {test_rmse}')
print(f'Test MAE: {test_mae}')