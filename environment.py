import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler

class TradingEnvironment:
    SELL = 0
    HOLD = 1
    BUY  = 2

    def __init__(
        self,
        ticker        = "SPY",
        start         = "2010-01-01",
        end           = "2022-12-31",
        window_size   = 20,
        initial_cash  = 10_000.0,
        transaction_cost = 0.001,
        train         = True,
        train_ratio   = 0.8,
    ):
        self.ticker           = ticker
        self.window_size      = window_size
        self.initial_cash     = initial_cash
        self.transaction_cost = transaction_cost


        raw = self._download(ticker, start, end)
        features = self._build_features(raw)
        self.prices = raw["Close"].reindex(features.index).reset_index(drop=True)

        


        split = int(len(features) * train_ratio)
        if train:
            self.data = features.iloc[:split].reset_index(drop=True)
        else:
            self.data = features.iloc[split:].reset_index(drop=True)

        
        self.n_features = self.data.shape[1]
        self.state_size = self.window_size * self.n_features + 2
        self.action_size = 3

        self.current_step  = None
        self.cash          = None
        self.shares_held   = None
        self.net_worth     = None
        self.prev_net_worth = None

    def _download(self, ticker, start, end):
        print(f"Downloading {ticker} from {start} to {end}...")

        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)
        print(f"  Downloaded {len(df)} trading days.")
        return df
    
    def _build_features(self, df):

        
        out = pd.DataFrame(index=df.index)

        close  = df["Close"]
        volume = df["Volume"]

        
        out["return_1d"] = close.pct_change().clip(-0.5, 0.5)

        out["return_5d"]  = close.pct_change(5).clip(-1, 1)
        out["return_10d"] = close.pct_change(10).clip(-1, 1)

        out["rsi_14"] = self._rsi(close, 14) / 100.0

        
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        out["macd"] = (ema_fast - ema_slow) / close


        macd_raw = ema_fast - ema_slow
        out["macd_signal"] = (macd_raw.ewm(span=9, adjust=False).mean()) / close

        rolling_mean = close.rolling(window=20).mean()
        rolling_std  = close.rolling(window=20).std()
        out["bb_position"] = ((close - rolling_mean) / (rolling_std + 1e-9)).clip(-3, 3)

        vol_mean = volume.rolling(window=20).mean()
        out["volume_ratio"] = (volume / (vol_mean + 1e-9)).clip(0, 5)

        out.dropna(inplace=True)

        scaler = RobustScaler()
        scaled = scaler.fit_transform(out.values)
        out = pd.DataFrame(scaled, columns=out.columns, index=out.index)

        print(f"  Built {out.shape[1]} features, {len(out)} usable rows after dropping NaN.")
        return out
    
    def _rsi(self, series, period=14):
        

        delta = series.diff()


        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)


        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()


        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))


    # ── reset() ───────────────────────────────────────────────────────────────
    def reset(self):
        self.current_step   = self.window_size
        self.cash           = self.initial_cash
        self.shares_held    = 0.0
        self.net_worth      = self.initial_cash
        self.prev_net_worth = self.initial_cash

        return self._get_state()


    # ── step() ────────────────────────────────────────────────────────────────
    def step(self, action):
        
        current_price = self._current_price()


        if action == self.BUY:
            shares_to_buy = self.cash / (current_price * (1 + self.transaction_cost))
            self.shares_held += shares_to_buy
            self.cash = 0.0

        elif action == self.SELL:
            proceeds = self.shares_held * current_price * (1 - self.transaction_cost)
            self.cash += proceeds
            self.shares_held = 0.0

        
        self.current_step += 1

        # ── Compute new net worth at the NEW step's price ─────────────────────
        new_price = self._current_price()
        self.net_worth = self.cash + self.shares_held * new_price

        # ── Reward ────────────────────────────────────────────────────────────
        
        reward = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        self.prev_net_worth = self.net_worth

        # ── Terminal condition ────────────────────────────────────────────────
        
        done = (
            self.current_step >= len(self.data) - 1
            or self.net_worth < self.initial_cash * 0.1
        )

        # ── Return ────────────────────────────────────────────────────────────
        # next_state: what the agent sees AFTER this step (tomorrow's view)
        # reward:     how good/bad this action was
        # done:       whether the episode is over
        next_state = self._get_state()
        return next_state, reward, done


    # ── _get_state() ──────────────────────────────────────────────────────────
    def _get_state(self):
        """
        Build the state vector the agent observes at the current timestep.

        The state is a flat 1D numpy array containing:
          [window_size days of features..., cash_ratio, position_ratio]

        Flattening is important — the neural network expects a 1D input vector,
        not a 2D matrix.
        """
        
        window = self.data.iloc[
            self.current_step - self.window_size : self.current_step
        ].values  


        flat = window.flatten()

        
        cash_ratio     = self.cash / self.initial_cash
        position_ratio = (self.shares_held * self._current_price()) / (self.net_worth + 1e-9)

        state = np.concatenate([flat, [cash_ratio, position_ratio]])
        return state.astype(np.float32)


    # ── _current_price() ──────────────────────────────────────────────────────
    def _current_price(self):
        
        return float(self.prices.iloc[self.current_step])


    # ── render() ──────────────────────────────────────────────────────────────
    def render(self):
        
        price = self._current_price()
        print(
            f"Step {self.current_step:4d} | "
            f"Price: ${price:8.2f} | "
            f"Cash: ${self.cash:10.2f} | "
            f"Shares: {self.shares_held:8.4f} | "
            f"Net Worth: ${self.net_worth:10.2f}"
        )

