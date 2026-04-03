import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler

class TradingEnvironment:
    SELL = 0
    HOLD = 1
    BUY  = 2