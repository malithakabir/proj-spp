
import numpy as np
from sklearn.metrics import mean_squared_error

def invert_scale_N_feature(scaler, X, yhat):
#   the 1st column is the 'Adj Close' price column
    result = np.zeros([X.shape[0]+yhat.shape[0], X.shape[1]])
    result[:X.shape[0], :] = X
    result[-yhat.shape[0]:, 0] = yhat
    
    inverted = scaler.inverse_transform(result)
    
    return inverted[:-yhat.shape[0], 0], inverted[-yhat.shape[0]:, 0]

