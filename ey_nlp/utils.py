# -*- coding: utf-8 -*-

def dense_identity(X):
    try:
        return X.todense()
    except:
        return X
    
def drop_column(X):
    ones = np.ones(X.shape[0])
    return ones[:, np.newaxis]