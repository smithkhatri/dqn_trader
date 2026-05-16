import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler


class TradingEnvironment:
    # ─────────────────────────────────────────────────────────────────────────
    # These are class-level constants. They live on the class itself, not on
    # any specific object. We use ALL_CAPS to signal they never change.
    # We represent actions as integers because the neural network outputs
    # a score for each integer index — 0, 1, 2.
    # ─────────────────────────────────────────────────────────────────────────
    SELL = 0
    HOLD = 1
    BUY  = 2

    def __init__(
        self,
        ticker           = "SPY",
        start            = "2010-01-01",
        end              = "2022-12-31",
        window_size      = 20,
        initial_cash     = 10_000.0,
        transaction_cost = 0.001,
        train            = True,
        train_ratio      = 0.8,
    ):
        # ─────────────────────────────────────────────────────────────────────
        # __init__ runs automatically when you do TradingEnvironment().
        # Its job is to receive all configuration and store it on the object
        # using self, so every other method can access it later.
        # Without self.ticker = ticker, the value would be lost the moment
        # __init__ finishes — it would die as a local variable.
        # ─────────────────────────────────────────────────────────────────────
        self.ticker           = ticker
        self.window_size      = window_size
        self.initial_cash     = initial_cash
        self.transaction_cost = transaction_cost
        self.train            = train
        self.train_ratio      = train_ratio

        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Download raw OHLCV data from Yahoo Finance.
        # We call our own helper method _download() to keep __init__ clean.
        # raw is a pandas DataFrame with columns: Open, High, Low, Close, Volume
        # ─────────────────────────────────────────────────────────────────────
        raw = self._download(ticker, start, end)

        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Build technical indicator features from the raw data.
        # features is a new DataFrame where each column is a meaningful signal
        # the agent can learn from — momentum, trend, volatility, volume.
        # ─────────────────────────────────────────────────────────────────────
        features = self._build_features(raw)

        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Store the raw closing prices aligned to the feature rows.
        # We need real dollar prices to execute trades (buy/sell at real price).
        # features has fewer rows than raw because indicators need warmup days
        # (e.g. a 20-day moving average needs 20 days before it produces a value).
        # We reindex so prices align perfectly with features, then reset the
        # integer index so .iloc[0] is always the first usable day.
        # ─────────────────────────────────────────────────────────────────────
        self.prices = raw["Close"].reindex(features.index).reset_index(drop=True)

        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Train/test split — ALWAYS by time, never randomly.
        # Random splitting would let the agent train on 2018 data and test on
        # 2015 data — that's cheating because the future informed the past.
        # We take the first 80% of rows for training, last 20% for testing.
        # reset_index(drop=True) resets row numbers to start from 0 in both
        # slices so .iloc indexing works cleanly in both modes.
        # ─────────────────────────────────────────────────────────────────────
        split = int(len(features) * train_ratio)
        if train:
            self.data = features.iloc[:split].reset_index(drop=True)
            self.prices = self.prices.iloc[:split].reset_index(drop=True)
        else:
            self.data = features.iloc[split:].reset_index(drop=True)
            self.prices = self.prices.iloc[split:].reset_index(drop=True)

        # ─────────────────────────────────────────────────────────────────────
        # Step 5: Compute and store dimensions the rest of the code needs.
        #
        # n_features: how many indicator columns exist in our feature dataframe.
        #
        # state_size: the total length of the vector we hand to the neural net.
        #   = window_size days × n_features per day, PLUS 2 extra values
        #   (cash_ratio and position_ratio — the agent's own portfolio state).
        #   The agent needs to know its own position so it doesn't try to sell
        #   when it holds nothing, or buy when it has no cash.
        #
        # action_size: always 3 — SELL, HOLD, BUY.
        # ─────────────────────────────────────────────────────────────────────
        self.n_features  = self.data.shape[1]
        self.state_size  = self.window_size * self.n_features + 2
        self.action_size = 3

        # ─────────────────────────────────────────────────────────────────────
        # Step 6: Define episode state variables.
        # These are set to None now — reset() will give them real values.
        # We define them here so Python knows they exist on the object.
        # This is a clean pattern: __init__ defines structure, reset() sets
        # starting values for each new episode.
        # ─────────────────────────────────────────────────────────────────────
        self.current_step   = None
        self.cash           = None
        self.shares_held    = None
        self.net_worth      = None
        self.prev_net_worth = None

        print(f"Environment ready — {len(self.data)} rows, state size: {self.state_size}")


    def _download(self, ticker, start, end):
        # ─────────────────────────────────────────────────────────────────────
        # The underscore prefix is a Python convention meaning "internal helper
        # — not meant to be called from outside the class."
        #
        # yf.download pulls historical OHLCV data from Yahoo Finance.
        # auto_adjust=True: adjusts historical prices for stock splits and
        #   dividends so prices are comparable across the entire history.
        #   Without it you'd see sudden price jumps that never really happened.
        # progress=False: silences the download progress bar. Fine once, but
        #   annoying if the environment is created many times during training.
        # ─────────────────────────────────────────────────────────────────────
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        # ─────────────────────────────────────────────────────────────────────
        # Newer versions of yfinance return a MultiIndex column structure like
        # ("Close", "SPY") instead of just "Close". We flatten it to simple
        # column names so the rest of the code works cleanly.
        # isinstance() checks if df.columns is a MultiIndex type.
        # get_level_values(0) keeps only the first level — "Close", not "SPY".
        # ─────────────────────────────────────────────────────────────────────
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)
        print(f"  {len(df)} trading days downloaded.")
        return df


    def _build_features(self, df):
        # ─────────────────────────────────────────────────────────────────────
        # This is the most important method after step().
        # Raw prices are useless to the neural network — a price of $450 means
        # nothing without context. We need scale-free signals that describe
        # what the market is DOING, not what it IS.
        #
        # Every single indicator here uses ONLY past data — no indicator peeks
        # even one day into the future. This is how we prevent lookahead bias
        # at the feature level.
        # ─────────────────────────────────────────────────────────────────────
        out   = pd.DataFrame(index=df.index)
        close  = df["Close"]
        volume = df["Volume"]

        # ─── 1. Returns ───────────────────────────────────────────────────────
        # pct_change() computes (today - yesterday) / yesterday — the daily
        # percentage move. This is a velocity signal: how fast is price moving
        # and in which direction?
        # clip(-0.5, 0.5) removes extreme outliers like data errors or splits
        # that slipped through. A 50% daily move is almost certainly bad data.
        # 5-day and 10-day returns give medium-term momentum context.
        out["return_1d"]  = close.pct_change().clip(-0.5, 0.5)
        out["return_5d"]  = close.pct_change(5).clip(-1, 1)
        out["return_10d"] = close.pct_change(10).clip(-1, 1)

        # ─── 2. RSI ───────────────────────────────────────────────────────────
        # Relative Strength Index — measures whether a stock has been going up
        # or down TOO much lately. Ranges 0 to 100.
        # Above 70 = overbought (might drop soon).
        # Below 30 = oversold (might rise soon).
        # We divide by 100 to normalize it to 0–1 range.
        out["rsi_14"] = self._rsi(close, 14) / 100.0

        # ─── 3. MACD ──────────────────────────────────────────────────────────
        # Moving Average Convergence Divergence.
        # Fast EMA(12) tracks recent price closely.
        # Slow EMA(26) tracks longer-term trend.
        # MACD = fast - slow. Positive = uptrend, negative = downtrend.
        # We divide by close to make it scale-free (same signal regardless of
        # whether the stock is $5 or $500).
        # EWM = Exponential Weighted Mean. adjust=False uses the recursive
        # formula which is standard for financial EMA calculations.
        ema_fast          = close.ewm(span=12, adjust=False).mean()
        ema_slow          = close.ewm(span=26, adjust=False).mean()
        macd_line         = ema_fast - ema_slow
        out["macd"]       = macd_line / close
        # Signal line = EMA(9) of MACD. When MACD crosses above signal = bullish.
        out["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean() / close

        # ─── 4. Bollinger Band position ───────────────────────────────────────
        # Measures how extreme the current price is relative to recent history.
        # We compute a z-score: (price - 20d_mean) / 20d_std.
        # 0 = right at the mean. +2 = two standard deviations above (expensive).
        # -2 = two standard deviations below (cheap).
        # The +1e-9 prevents division by zero on days with zero variance.
        # clip(-3, 3) keeps it from exploding during extreme market events.
        rolling_mean      = close.rolling(20).mean()
        rolling_std       = close.rolling(20).std()
        out["bb_position"] = ((close - rolling_mean) / (rolling_std + 1e-9)).clip(-3, 3)

        # ─── 5. Volume ratio ──────────────────────────────────────────────────
        # Today's volume divided by its 20-day average.
        # 1.0 = average activity. 2.0 = twice the usual volume.
        # High volume often precedes or confirms big price moves.
        # clip(0, 5) caps extreme outliers — a volume 5x the average is already
        # a very strong signal, anything higher is noise.
        vol_mean           = volume.rolling(20).mean()
        out["volume_ratio"] = (volume / (vol_mean + 1e-9)).clip(0, 5)

        # ─────────────────────────────────────────────────────────────────────
        # Drop NaN rows. Rolling windows need N past days before producing
        # a value. The first ~26 rows will be NaN for the longer indicators.
        # dropna() removes any row that has at least one NaN anywhere.
        # ─────────────────────────────────────────────────────────────────────
        out.dropna(inplace=True)

        # ─────────────────────────────────────────────────────────────────────
        # Scale with RobustScaler.
        # Even after all our normalization, features are on different scales.
        # Returns are tiny decimals. Volume ratio is 0–5. RSI is 0–1.
        # Neural networks learn much faster when all inputs are on a similar
        # scale — otherwise large-valued features dominate the gradients.
        #
        # Why RobustScaler and not StandardScaler?
        # StandardScaler uses mean and standard deviation — both are heavily
        # influenced by outliers (crashes, spikes). Financial data has many
        # outliers. RobustScaler uses median and interquartile range instead,
        # which are resistant to outliers. Much more stable for market data.
        #
        # fit_transform() computes the scaling parameters AND applies them
        # in one step. It returns a numpy array, so we wrap it back into a
        # DataFrame with the same column names and index.
        # ─────────────────────────────────────────────────────────────────────
        scaler = RobustScaler()
        scaled = scaler.fit_transform(out.values)
        out    = pd.DataFrame(scaled, columns=out.columns, index=out.index)

        print(f"  {out.shape[1]} features built, {len(out)} usable rows.")
        return out


    def _rsi(self, series, period=14):
        # ─────────────────────────────────────────────────────────────────────
        # Manual RSI computation. Understanding this matters more than importing
        # a library that does it for you.
        #
        # Step 1: compute day-over-day change.
        # Step 2: separate into gains (positive changes) and losses (negatives
        #   made positive — we want the magnitude, not the sign).
        # Step 3: compute the exponential moving average of gains and losses.
        #   com=period-1 sets the center-of-mass for the EWM, which matches
        #   Wilder's original smoothing formula used in standard RSI.
        #   min_periods=period means: don't output a value until we have
        #   enough data for a meaningful average.
        # Step 4: RS = avg_gain / avg_loss. RSI = 100 - 100/(1+RS).
        #   +1e-9 prevents division by zero when avg_loss is exactly 0
        #   (happens during long uninterrupted rallies).
        # ─────────────────────────────────────────────────────────────────────
        delta    = series.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))


    def reset(self):
        # ─────────────────────────────────────────────────────────────────────
        # reset() starts a fresh episode. Called once at the start of training,
        # then again after every episode ends (when done=True from step()).
        #
        # Why start at window_size instead of 0?
        # The agent's first observation needs a full window of past data.
        # If we started at 0, _get_state() would try to look back window_size
        # days before the data begins — nothing is there.
        # Starting at window_size guarantees the first lookback window is full.
        # ─────────────────────────────────────────────────────────────────────
        self.current_step   = self.window_size
        self.cash           = self.initial_cash
        self.shares_held    = 0.0
        self.net_worth      = self.initial_cash
        self.prev_net_worth = self.initial_cash
        return self._get_state()


    def step(self, action):
        # ─────────────────────────────────────────────────────────────────────
        # step() is the core of the environment. The agent calls this every day
        # with its chosen action. We execute the trade, advance time, compute
        # reward, and return what the agent needs to keep learning.
        #
        # Returns: (next_state, reward, done)
        # This is the universal RL interface — every RL environment on earth
        # follows this exact pattern.
        # ─────────────────────────────────────────────────────────────────────

        # Get today's price BEFORE advancing time.
        # This is the price at which we execute the trade.
        # If we advanced time first, we'd be trading at tomorrow's price —
        # that's lookahead bias in the step logic.
        current_price = self._current_price()

        # ─── Execute the trade ────────────────────────────────────────────────
        if action == self.BUY:
            # Spend ALL available cash. We go fully invested each time.
            # Dividing by (1 + cost) accounts for the broker fee —
            # the effective price we pay per share is slightly above market.
            shares_to_buy    = self.cash / (current_price * (1 + self.transaction_cost))
            self.shares_held += shares_to_buy
            self.cash         = 0.0

        elif action == self.SELL:
            # Sell ALL shares. Full liquidation each time.
            # Multiplying by (1 - cost) accounts for the broker fee —
            # the effective proceeds we receive are slightly below market.
            proceeds         = self.shares_held * current_price * (1 - self.transaction_cost)
            self.cash       += proceeds
            self.shares_held = 0.0

        # HOLD: do nothing. Cash and shares stay exactly as they are.

        # ─── Advance time ────────────────────────────────────────────────────
        # We move to the next day AFTER executing the trade at today's price.
        self.current_step += 1

        # ─── Compute new net worth ────────────────────────────────────────────
        # Net worth = cash on hand + value of shares at TODAY's new price.
        new_price      = self._current_price()
        self.net_worth = self.cash + self.shares_held * new_price

        # ─── Compute reward ───────────────────────────────────────────────────
        # Reward = percentage change in portfolio value since last step.
        # We use percentage change (not raw dollar change) so the reward scale
        # stays consistent regardless of how much the portfolio has grown.
        # This is called a differential reward — we reward improvement,
        # not absolute wealth.
        reward              = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        self.prev_net_worth = self.net_worth

        # ─── Check if episode is done ─────────────────────────────────────────
        # Done when we run out of data, OR if the agent loses 90% of capital.
        # The second condition stops a failing episode early — no point
        # continuing when the agent has already blown up.
        done = (
            self.current_step >= len(self.data) - 1
            or self.net_worth < self.initial_cash * 0.1
        )

        return self._get_state(), reward, done


    def _get_state(self):
        # ─────────────────────────────────────────────────────────────────────
        # Build the state vector — the snapshot of the world the agent sees.
        #
        # It contains two things:
        # 1. A window of market data: the last window_size days of features,
        #    flattened into a 1D array. Shape: (window_size * n_features,)
        # 2. Portfolio state: cash_ratio and position_ratio — two numbers
        #    that tell the agent what its own financial position currently is.
        #
        # Why flatten? The neural network expects a 1D vector as input.
        # A 2D window (20 days × 8 features) becomes a 1D array of 160 values.
        # Then we append 2 portfolio numbers → total length 162.
        # ─────────────────────────────────────────────────────────────────────

        # Grab the last window_size rows ending at current_step.
        # iloc uses integer positions. [start:end] in pandas excludes end.
        window = self.data.iloc[
            self.current_step - self.window_size : self.current_step
        ].values  # .values converts DataFrame to numpy array

        # Flatten from (window_size, n_features) to (window_size * n_features,)
        flat = window.flatten()

        # cash_ratio: how much of starting capital is still in cash?
        # position_ratio: what fraction of current net worth is in shares?
        # These tell the agent its own position without it having to infer it.
        cash_ratio     = self.cash / self.initial_cash
        position_ratio = (self.shares_held * self._current_price()) / (self.net_worth + 1e-9)

        # Concatenate everything into one flat vector.
        # np.concatenate joins arrays end to end.
        # [cash_ratio, position_ratio] wraps the two scalars in a list so
        # concatenate sees them as a 1D array to append.
        # float32 is what PyTorch tensors expect — float64 causes type errors.
        state = np.concatenate([flat, [cash_ratio, position_ratio]])
        return state.astype(np.float32)


    def _current_price(self):
        # ─────────────────────────────────────────────────────────────────────
        # Returns the closing price at the current step.
        # self.prices is the raw (unscaled) close price Series we stored in
        # __init__. We use real dollar prices for trade execution — not the
        # scaled feature values which have no dollar meaning.
        # float() converts a single-element pandas value to a plain Python float.
        # ─────────────────────────────────────────────────────────────────────
        return float(self.prices.iloc[self.current_step])


    def render(self):
        # ─────────────────────────────────────────────────────────────────────
        # Prints a human-readable snapshot of the current state.
        # Useful for debugging — you can call this inside the training loop
        # to watch the agent trade day by day.
        # ─────────────────────────────────────────────────────────────────────
        p = self._current_price()
        print(
            f"Step {self.current_step:4d} | "
            f"Price ${p:8.2f} | "
            f"Cash ${self.cash:10.2f} | "
            f"Shares {self.shares_held:8.4f} | "
            f"Net Worth ${self.net_worth:10.2f}"
        )