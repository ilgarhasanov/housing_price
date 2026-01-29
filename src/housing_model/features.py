import numpy as np
from sklearn.preprocessing import FunctionTransformer

def safe_ratio(X: np.ndarray) -> np.ndarray:
    num = X[:, [0]].astype(np.float64)
    den = X[:, [1]].astype(np.float64)
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=(den!=0))
    return out # iki susutnu bolur

def ratio_name(transformer, feature_names_in):
    return ['ratio']

#
ratio_transformer = FunctionTransformer(safe_ratio, feature_names_out = ratio_name)

def safe_log1p(X: np.array) -> np.array:
    X = X.astype(np.float64)
    X = np.clip(X, a_min = 0, a_max=None)
    return np.log1p(X) #np.log(1+X) 1 gelir cunki log(0) infinite qaytarir. ona gore 1 gelirik 

log_transformer = FunctionTransformer(
    safe_log1p,
    inverse_func=np.expm1,
    feature_names_out = 'one-to-one'
)