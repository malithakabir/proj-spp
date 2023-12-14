
import math
import numpy as np
from sklearn.metrics import mean_squared_error as mse

def invert_scale_N_feature(scaler, X, yhat):
#   the 1st column is the 'Adj Close' price column
    result = np.zeros([X.shape[0]+yhat.shape[0], X.shape[1]])
    result[:X.shape[0], :] = X
    result[-yhat.shape[0]:, 0] = yhat
    
    inverted = scaler.inverse_transform(result)
    
    return inverted[:-yhat.shape[0], 0], inverted[-yhat.shape[0]:, 0]

def calculate_rmse(scalers, data, labels, predicts, splitname):
    # invert the predicted and original test value to USD
    
    invert_predict = []     # predicted future test values (hold_days)
    invert_true = []        # true future test values (hold_days)
    actual = []             # actual test data in the previous days (seq_len)
    
    for i in range(len(predicts)):
    #    for i in range(1):
       _actual, inverted_pred = invert_scale_N_feature(scalers[i], data[i,:,:], predicts[i])
       invert_predict.append(inverted_pred)
       
       _actual, inverted_true = invert_scale_N_feature(scalers[i], data[i,:,:], labels[i])
       invert_true.append(inverted_true)
       actual.append(_actual)
    
    total_rmse = 0
    for i in range(len(invert_predict)):
       total_rmse = total_rmse + math.sqrt(mse(invert_predict[i], invert_true[i]))
    accuracy = total_rmse/len(invert_predict)
    print("The avearge root mse on %s data is: %.2f" %(splitname, accuracy))
    return accuracy
    