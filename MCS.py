import numpy as np
from scipy import stats

# Example MSE values for different models at different times
# Rows are models (Model 1, Model 2, etc.) and columns are times (t1, t2, t3, etc.)
# Replace these with your actual MSE values (L values).
def MCS(L):
    # Number of models
    best_model = np.argmin(np.mean(L, axis=1))
    model_dis = L - L[best_model]

    M = L.shape[0]
    alpha = 0.05
    p_values = np.full(M, np.nan)
    included_models = np.zeros(M)
    included_models[best_model] = 1  # Keep track of models to be included in the SSM

    for i in range(M):
        if i != best_model:
            # Perform a t-test to compare models
            t_stat, p_value = stats.ttest_1samp(model_dis[i], 0)  # Test if mean difference is 0
            p_values[i] = p_value  # Store p-value
            if p_value > alpha:
                included_models[i] = 1

    return {"modelindex": included_models,
            "P_values": p_values}