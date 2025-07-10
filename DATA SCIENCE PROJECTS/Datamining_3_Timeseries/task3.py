import numpy as np


def create_lagged_features(x, exogenous, lag=3):
    # x is a (n_timestep, ) array
    # exogenous is an (n_timestep, n_variables) array
    X, y = [], []
    for i in range(len(x) - lag):
        # TODO: ensure that the data contains also all exogenous variables.
        # 'concatenate' the lagged feature and the exogenous features.
        lagged_features = x[i:i+lag]
        exogenous_features = exogenous[i]
        X.append(np.concatenate((lagged_features,exogenous_features)))
        y.append(x[i + lag])

    return np.asarray(X), np.asarray(y)


def iterative_forecast(reg, last_known, last_known_exogenous, steps):
    # last_known_exogenous is the last known exogenous variable of the training data
    # last_known is the last_known lagged training sample
    forecast = []
    window = list(last_known)
    for _ in range(steps):
        # TODO: ensure the predict function also gets the exogenous variables
        # Remember that they should occupy the same position in the feature
        # vector
        data = np.concatenate((window,last_known_exogenous))
        predicted = reg.predict([data])[0]
        window.append(predicted)
        forecast.append(predicted)
        window.pop(0)

    return np.array(forecast)