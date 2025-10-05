
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from sklearn.preprocessing import MinMaxScaler


# ---------- Streamlit Setup ----------
st.set_page_config("Market Forecasting (LR, RF, XGB, LSTM)", layout="wide")
st.title("📈 Market Prediction using Supervised Machine Learning Models")

tabs = st.tabs([
    "Raw Data",
    "📈 Multiple Linear Regression",
    "🌲 Random Forest",
    "⚡ XGBoost",
    "🧬 Deep Learning (LSTM)",
    "📉 NIFTY Constituents",
    "🏦 BankNIFTY Constituents"
    
])

# ---------- Sidebar ----------
tickers_list = [
    # Indian
    'RELIANCE.NS','TCS.NS','INFY.NS','HDFCBANK.NS','ICICIBANK.NS','LT.NS','SBIN.NS',
    '^NSEI','^NSEBANK',
    # US
    'AAPL','MSFT','GOOG','TSLA','AMZN','NVDA','META',
    # Crypto
    'BTC-USD','ETH-USD'
]

with st.sidebar:
    st.markdown("### 📈 Asset Symbol Options")

    ticker_input = st.text_input("Type your ticker:", "AAPL", key="ticker_input").upper()
    ticker_options = [ticker_input] + [t for t in tickers_list if t != ticker_input]
    ticker_symbol = st.selectbox("Or select any ticker from list:", ticker_options, index=0)

    interval_options = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
    selected_interval = st.selectbox("Select Time Interval:", interval_options, index=3)

    forecast_periods = st.slider("Forecast Periods (future points)", 1, 50, 5)

    run = st.button("▶️ Run Forecast For Asset")

    st.write("Ticker:", ticker_symbol)
    st.write("Interval:", selected_interval)
    st.write("Forecast Periods:", forecast_periods)


# ---------- Data Fetch ----------
def fetch_data(ticker, interval):
    if interval in ["1m", "2m", "5m","15m"]:
        period = "5d"
    elif interval in ["30m",  "60m"]:
    
        period = "1mo"
    elif interval in [ "90m", "1h"]:
        period = "6mo"
    
    elif interval == "1d":
        period = "1y"
    elif interval == "5d":
        interval = "1d"; period = "5y"
    elif interval == "1wk":
        period = "10y"
    elif interval in ["1mo", "3mo"]:
        period = "20y"
    else:
        period = "1y"

    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df = df.reset_index()
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "date"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    return df


def interval_to_offset(interval: str):
    """Map Yahoo interval string to pandas offset"""
    mapping = {
        "1m": "1min",
        "2m": "2min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "90m": "90min",
        "1h": "1h",
        "1d": "1d",
        "5d": "5d",
        "1wk": "7d",
        "1mo": "30d",
        "3mo": "90d"
    }
    return mapping.get(interval, "1d")




def feature_engineering(df):
    df = df.copy()

    # If not enough rows, bail out early
    if len(df) < 20:
        return pd.DataFrame()

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # --- Volume-based features (guard for indices with no volume) ---
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume']
        ).on_balance_volume()
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            high=df['High'], low=df['Low'], close=df['Close'],
            volume=df['Volume'], window=20
        )
        df['cmf'] = cmf.chaikin_money_flow()
    else:
        df['obv'] = 0.0
        df['cmf'] = 0.0

    # Moving averages
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['ema_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_high'] - df['bb_low']

    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # ATR (use 7-period so less data required)
    if len(df) >= 7:
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=7
        ).average_true_range()
    else:
        df['atr'] = np.nan

    # Momentum / ROC
    df['momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['roc_10'] = ta.momentum.ROCIndicator(close=df['Close'], window=10).roc()

    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], lbp=14
    ).williams_r()

    # CCI
    df['cci'] = ta.trend.CCIIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], window=20
    ).cci()

    # Lags
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    # Log return
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Shift non-lag features to prevent leakage
    exclude = {'date', 'Open', 'High', 'Low', 'Close', 'Volume',
               'lag_1','lag_2','lag_3','lag_4','lag_5'}
    feature_cols = [c for c in df.columns if c not in exclude]
    for col in feature_cols:
        df[col] = df[col].shift(1)

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ---------- Utilities ----------

def train_test_split(df):
    if df is None or df.empty or "Close" not in df.columns:
        return None, None, None, None, None, []
    
    features = [c for c in df.columns if c not in ['date','Close']]
    X, y = df[features], df['Close']

    if len(X) < 5:
        return None, None, None, None, None, features

    split = int(len(X) * 0.7)
    if split == 0 or split >= len(X):
        return None, None, None, None, None, features

    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], df.iloc[split:]['date'], features


# def train_test_split(df):
#     features = [c for c in df.columns if c not in ['date','Close']]
#     X, y = df[features], df['Close']
#     split = int(len(X)*0.7)
#     return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], df.iloc[split:]['date'], features

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)   # manual RMSE so no 'squared' arg
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape



# ---------- Raw Data ----------
# ================================
# Part 2: Model Tabs (LR, RF, XGB, LSTM)
# ================================

# ---------- Raw Data ----------
with tabs[0]:
    st.header("📊 Raw Data")
    if run:
        df = fetch_data(ticker_symbol, selected_interval)
        st.subheader("Recent Data")
        st.dataframe(df.tail(10))

        df_feat = feature_engineering(df)
        st.subheader("Features (engineered)")
        st.dataframe(df_feat.tail(10))

# ---------- Multiple Linear Regression ----------
with tabs[1]:
    st.header("📈 Multiple Linear Regression")
    if run:
        df = feature_engineering(fetch_data(ticker_symbol, selected_interval))
        X_train, X_test, y_train, y_test, test_dates, features = train_test_split(df)

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae, rmse, mape = evaluate(y_test, preds)
        c1,c2,c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")

        # Actual vs Predicted
        df_results = pd.DataFrame({"Date": test_dates, "Actual": y_test, "Predicted": preds})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_results["Date"], y=df_results["Actual"], mode="lines", name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df_results["Date"], y=df_results["Predicted"], mode="lines", name="Predicted", line=dict(color="orange", dash="dot")))
        fig.update_layout(title=f"{ticker_symbol} - Linear Regression", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Residuals
        df_results["Residuals"] = df_results["Actual"] - df_results["Predicted"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_results["Date"], y=df_results["Residuals"], mode="lines", name="Residuals", line=dict(color="purple")))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)

        # Coefficients
        coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
        coef_df["abs"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("abs", ascending=False)
        st.subheader("Top Coefficients")
        st.dataframe(coef_df.head(20))

      


        st.subheader("📌 Next Predictions")

        future_preds = []
        future_dates = []
        temp_df = df.copy()

        last_date = temp_df["date"].iloc[-1]
        offset = interval_to_offset(selected_interval)
        for i in range(forecast_periods):
            # use most recent features
            last_features = temp_df[features].iloc[[-1]]
            pred = model.predict(last_features)[0]

            #next_date = last_date + pd.to_timedelta(1, unit=selected_interval if selected_interval not in ["1d","1wk","1mo"] else "d")
            next_date = last_date + pd.to_timedelta(offset)

            future_preds.append(pred)
            future_dates.append(next_date)

            # append new row with updated Close
            new_row = temp_df.iloc[[-1]].copy()
            new_row["date"] = next_date
            new_row["Close"] = pred
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)

            # recompute lag features for the new row
            for lag in range(1, 6):
                temp_df.loc[temp_df.index[-1], f"lag_{lag}"] = temp_df["Close"].iloc[-lag-1]

            last_date = next_date

        # one consolidated DataFrame
        df_future = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": future_preds
        })

        st.dataframe(df_future)
    else:
        st.info("""
Enter Stock/Crypto/Commodity Symbol from Yahoo Finance (e.g., US Stocks: `AAPL`, `MSFT`; Indian Stocks: `RELIANCE.NS`, `TCS.NS`; Crypto: `BTC-USD`, `ETH-USD`; Indices: `^NSEI`, `^DJI`; Commodities: `GC=F`, `CL=F`).  
Click ▶️ Run Forecast to see predictions (typing a ticker overrides dropdown selection).
""")



# ---------- Random Forest ----------
with tabs[2]:
    st.header("🌲 Random Forest (Auto-Tuned)")
    if run:
        df = feature_engineering(fetch_data(ticker_symbol, selected_interval))
        X_train, X_test, y_train, y_test, test_dates, features = train_test_split(df)

        # Hyperparameter tuning (quick search)
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": [200, 300, 500],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        search = RandomizedSearchCV(
            rf_base,
            param_distributions=param_dist,
            n_iter=5, cv=3, n_jobs=-1,
            scoring="neg_mean_squared_error",
            random_state=42
        )
        with st.spinner("Tuning RandomForest..."):
            search.fit(X_train, y_train)

        best_params = search.best_params_
        rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)

        # Metrics
        mae, rmse, mape = evaluate(y_test, preds)
        c1,c2,c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")

        # Actual vs Predicted
        df_results = pd.DataFrame({"Date": test_dates, "Actual": y_test, "Predicted": preds})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_results["Date"], y=df_results["Actual"],
                                 mode="lines", name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df_results["Date"], y=df_results["Predicted"],
                                 mode="lines", name="Predicted", line=dict(color="orange", dash="dot")))
        st.plotly_chart(fig, use_container_width=True)

        # Residuals
        df_results["Residuals"] = df_results["Actual"] - df_results["Predicted"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_results["Date"], y=df_results["Residuals"],
                                  mode="lines", name="Residuals", line=dict(color="green")))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)

        # Feature Importances
        imp = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_})
        imp = imp.sort_values("Importance", ascending=False)
        st.subheader("Feature Importances")
        st.dataframe(imp.head(20))

        fig3 = go.Figure(go.Bar(
            x=imp["Importance"].head(20),
            y=imp["Feature"].head(20),
            orientation="h"
        ))
        st.plotly_chart(fig3, use_container_width=True)

        # ---- Next Predictions ----
        st.subheader("📌 Next Predictions")

        def interval_to_offset(interval: str):
            mapping = {
                "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min",
                "30m": "30min", "60m": "60min", "90m": "90min", "1h": "1h",
                "1d": "1d", "5d": "5d", "1wk": "7d", "1mo": "30d", "3mo": "90d"
            }
            return mapping.get(interval, "1d")

        offset = interval_to_offset(selected_interval)

        future_preds, future_dates = [], []
        temp_df = df.copy()
        last_date = temp_df["date"].iloc[-1]

        for _ in range(forecast_periods):
            last_features = temp_df[features].iloc[[-1]]
            pred = rf.predict(last_features)[0]

            next_date = last_date + pd.to_timedelta(offset)
            future_preds.append(pred)
            future_dates.append(next_date)

            # append new row
            new_row = temp_df.iloc[[-1]].copy()
            new_row["date"] = next_date
            new_row["Close"] = pred
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)

            # 🔑 recompute all indicators so features update
            temp_df = feature_engineering(temp_df)

            last_date = next_date

        df_future = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_preds})
        st.dataframe(df_future)

    else:
        st.info("""
Enter Stock/Crypto/Commodity Symbol from Yahoo Finance (e.g., US Stocks: `AAPL`, `MSFT`; Indian Stocks: `RELIANCE.NS`, `TCS.NS`; Crypto: `BTC-USD`, `ETH-USD`; Indices: `^NSEI`, `^DJI`; Commodities: `GC=F`, `CL=F`).  
Click ▶️ Run Forecast to see predictions (typing a ticker overrides dropdown selection).
""")



with tabs[3]:
    st.header("⚡ XGBoost (Tuned)")
    if run:
        df = feature_engineering(fetch_data(ticker_symbol, selected_interval))
        result = train_test_split(df)

        if result[0] is None:
            st.warning("Not enough rows for training.")
        else:
            X_train, X_test, y_train, y_test, test_dates, features = result

            # Hyperparameter tuning
            xgb_base = XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42)
            param_dist = {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0]
            }

            search = RandomizedSearchCV(
                xgb_base,
                param_distributions=param_dist,
                n_iter=10, cv=3, n_jobs=-1,
                scoring="neg_mean_squared_error",
                random_state=42
            )
            with st.spinner("Tuning XGBoost..."):
                search.fit(X_train, y_train)

            best_params = search.best_params_
            st.write("**Best Parameters:**", best_params)

            xgb = XGBRegressor(**best_params, objective="reg:squarederror", n_jobs=-1, random_state=42)
            xgb.fit(X_train, y_train)
            preds = xgb.predict(X_test)

            # Metrics
            mae, rmse, mape = evaluate(y_test, preds)
            c1,c2,c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")
            c3.metric("MAPE", f"{mape:.2f}%")

            # Plots
            df_results = pd.DataFrame({"Date": test_dates, "Actual": y_test, "Predicted": preds})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_results["Date"], y=df_results["Actual"], mode="lines", name="Actual", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df_results["Date"], y=df_results["Predicted"], mode="lines", name="Predicted", line=dict(color="orange", dash="dot")))
            st.plotly_chart(fig, use_container_width=True)

            # ---- Fixed Future Predictions ----
            st.subheader("📌 Next Predictions")

            def interval_to_offset(interval: str):
                mapping = {
                    "1m": "1min","2m":"2min","5m":"5min","15m":"15min","30m":"30min",
                    "60m":"60min","90m":"90min","1h":"1h","1d":"1d","5d":"5d","1wk":"7d",
                    "1mo":"30d","3mo":"90d"
                }
                return mapping.get(interval, "1d")

            offset = interval_to_offset(selected_interval)

            future_preds, future_dates = [], []
            temp_df = df.copy()
            last_date = temp_df["date"].iloc[-1]

            for _ in range(forecast_periods):
                if temp_df.empty: break
                last_features = temp_df[features].iloc[[-1]]
                pred = xgb.predict(last_features)[0]

                next_date = last_date + pd.to_timedelta(offset)
                future_preds.append(pred)
                future_dates.append(next_date)

                new_row = temp_df.iloc[[-1]].copy()
                new_row["date"] = next_date
                new_row["Close"] = pred
                temp_df = pd.concat([temp_df, new_row], ignore_index=True)

                # update lag features manually
                for lag in range(1, 6):
                    if len(temp_df) > lag:
                        temp_df.loc[temp_df.index[-1], f"lag_{lag}"] = temp_df["Close"].iloc[-lag-1]

                last_date = next_date

            df_future = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_preds})
            st.dataframe(df_future)
    else:
        st.info("Enter a ticker and click ▶️ Run Forecast to see XGBoost results.")


from tensorflow.keras import backend as K


from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

# ---------- LSTM ----------
with tabs[4]:
    st.header(" Deep Learning (LSTM)")
    if run:
        df = feature_engineering(fetch_data(ticker_symbol, selected_interval))
        features = [c for c in df.columns if c not in ["date","Close"]]

        # Scale features + target together
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features + ["Close"]])

        lookback = 60
        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i, :-1])
            y.append(scaled[i, -1])
        X, y = np.array(X), np.array(y)

        if len(X) < 100:
            st.warning("Not enough data for LSTM at this interval.")
        else:
            split = int(0.7*len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Clear old session to avoid "NoneType pop" error
            K.clear_session()

            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.1),
                LSTM(64),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")

            # Early stopping
            es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

            model.fit(X_train, y_train,
                      validation_split=0.1,
                      epochs=50,
                      batch_size=32,
                      callbacks=[es],
                      verbose=0)

            # Predictions
            preds = model.predict(X_test)
            preds_rescaled = scaler.inverse_transform(
                np.concatenate([np.zeros((len(preds), len(features))), preds], axis=1)
            )[:,-1]
            y_test_rescaled = scaler.inverse_transform(
                np.concatenate([np.zeros((len(y_test), len(features))), y_test.reshape(-1,1)], axis=1)
            )[:,-1]

            # Metrics
            mae, rmse, mape = evaluate(y_test_rescaled, preds_rescaled)
            c1,c2,c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")
            c3.metric("MAPE", f"{mape:.2f}%")

            # Actual vs Predicted
            df_results = pd.DataFrame({"Actual": y_test_rescaled, "Predicted": preds_rescaled})
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df_results["Actual"], mode="lines", name="Actual", line=dict(color="blue")))
            fig.add_trace(go.Scatter(y=df_results["Predicted"], mode="lines", name="Predicted", line=dict(color="orange", dash="dot")))
            st.plotly_chart(fig, use_container_width=True)

            # Residuals
            df_results["Residuals"] = df_results["Actual"] - df_results["Predicted"]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=df_results["Residuals"], mode="lines", name="Residuals", line=dict(color="purple")))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig2, use_container_width=True)

            # Directional Accuracy
            df_results["Actual_Dir"] = df_results["Actual"].diff().apply(lambda x: 1 if x > 0 else -1)
            df_results["Pred_Dir"] = df_results["Predicted"].diff().apply(lambda x: 1 if x > 0 else -1)
            df_results.dropna(inplace=True)
            dir_acc = (df_results["Actual_Dir"] == df_results["Pred_Dir"]).mean() * 100
            st.metric("Directional Accuracy", f"{dir_acc:.2f}%")

            # ---- Next Predictions ----
            st.subheader("📌 Next Predictions")

            def interval_to_offset(interval: str):
                mapping = {
                    "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min",
                    "30m": "30min", "60m": "60min", "90m": "90min", "1h": "1h",
                    "1d": "1d", "5d": "5d", "1wk": "7d", "1mo": "30d", "3mo": "90d"
                }
                return mapping.get(interval, "1d")

            offset = interval_to_offset(selected_interval)

            last_seq = X_test[-1:]
            preds_future, future_dates = [], []
            last_date = df["date"].iloc[-1]

            for _ in range(forecast_periods):
                p = model.predict(last_seq)[0][0]

                inv_p = scaler.inverse_transform(
                    np.concatenate([np.zeros((1, len(features))), np.array(p).reshape(1,1)], axis=1)
                )[0,-1]
                preds_future.append(inv_p)

                next_date = last_date + pd.to_timedelta(offset)
                future_dates.append(next_date)

                # update sequence
                new_step = np.zeros((1,1,last_seq.shape[2]))
                new_step[0,0,-1] = p
                last_seq = np.concatenate([last_seq[:,1:,:], new_step], axis=1)

                last_date = next_date

            df_future = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds_future})
            st.dataframe(df_future)

    else:
        st.info("""
Enter Stock/Crypto/Commodity Symbol from Yahoo Finance (e.g., US Stocks: `AAPL`, `MSFT`; Indian Stocks: `RELIANCE.NS`, `TCS.NS`; Crypto: `BTC-USD`, `ETH-USD`; Indices: `^NSEI`, `^DJI`; Commodities: `GC=F`, `CL=F`).  
Click ▶️ Run Forecast to see predictions (typing a ticker overrides dropdown selection).
""")
# ================================
# Part 3: Constituents, Portfolio, Overview
# ================================

# ---------- Helper for batch prediction ----------
@st.cache_data
def batch_predict(tickers, interval, model_type, forecast_periods):
    results = {}
    for t in tickers:
        try:
            df = feature_engineering(fetch_data(t, interval))
            if len(df) < 10:
                continue
            X_train, X_test, y_train, y_test, test_dates, features = train_test_split(df)

            if model_type == "LR":
                model = LinearRegression()
            elif model_type == "RF":
                model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
            else:
                model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # take last forecast_periods values
            preds_tail = preds[-forecast_periods:].tolist()

            # add next prediction
            last_row = df[features].iloc[[-1]]
            next_pred = model.predict(last_row)[0]
            preds_tail.append(next_pred)

            results[t] = preds_tail
        except Exception as e:
            continue
    return pd.DataFrame(results)

# ---------- Constituents ----------
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","LT.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
    "HINDUNILVR.NS","ITC.NS","BHARTIARTL.NS","WIPRO.NS","TECHM.NS","ASIANPAINT.NS","MARUTI.NS","NESTLEIND.NS",
    "TITAN.NS","ULTRACEMCO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","ADANIENT.NS","ADANIPORTS.NS","ONGC.NS","COALINDIA.NS",
    "POWERGRID.NS","NTPC.NS","GRASIM.NS","JSWSTEEL.NS","TATASTEEL.NS","HCLTECH.NS","SUNPHARMA.NS","CIPLA.NS",
    "DRREDDY.NS","BRITANNIA.NS","HEROMOTOCO.NS","EICHERMOT.NS","HINDALCO.NS","DIVISLAB.NS","BAJAJ-AUTO.NS",
    "M&M.NS","BPCL.NS","IOC.NS","SHREECEM.NS","TATAMOTORS.NS","TATACONSUM.NS","APOLLOHOSP.NS","INDUSINDBK.NS","UPL.NS","SBILIFE.NS"
]

BANKNIFTY = [
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
    "INDUSINDBK.NS","PNB.NS","BANKBARODA.NS","AUBANK.NS","IDFCFIRSTB.NS","FEDERALBNK.NS","BANDHANBNK.NS"
]

# ---------- NIFTY Constituents ----------


# ---------- NIFTY Constituents ----------
with tabs[5]:
    st.header("📉 NIFTY Constituents Predictions & Correlation")

    model_choice = st.selectbox("Model", ["LR","RF","XGB"], index=1, key="nifty_model")

    if st.button("🔮 Predict"):
        with st.spinner("Running predictions... please wait"):
            # predict for constituents
            df_preds = batch_predict(NIFTY50, selected_interval, model_choice, forecast_periods)
            # predict for the index itself (you may need to adjust ticker name to your data source)
            df_nifty = batch_predict(["^NSEI"], selected_interval, model_choice, forecast_periods)

        if df_preds.empty or df_nifty.empty:
            st.warning("Not enough data to generate predictions.")
        else:
            # --- Show predictions ---
            st.subheader("Predicted Closes (last steps + next forecast)")

            # Add nifty into results for display
            df_preds_with_index = df_preds.copy()
            df_preds_with_index["NIFTY"] = df_nifty.iloc[:, 0]

            st.dataframe(
                df_preds_with_index.tail().T.reset_index().rename(columns={"index": "Ticker"})
            )

            # --- Pairwise correlation ---
            st.subheader("Correlation Heatmap (full predictions)")
            fig, ax = plt.subplots(figsize=(12,8))
            sns.heatmap(df_preds_with_index.corr(), cmap="RdBu_r", center=0,
                        vmin=-1, vmax=1, annot=False, ax=ax)
            st.pyplot(fig)

            # --- Correlation with NIFTY only ---
            st.subheader("Correlation with NIFTY predictions")
            corr_with_nifty = df_preds_with_index.corr()["NIFTY"].drop("NIFTY")

            corr_sorted = corr_with_nifty.sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(10,6))
            corr_sorted.plot(kind="bar", ax=ax2, color="teal")
            ax2.set_ylabel("Correlation")
            ax2.set_title("Ticker correlation with NIFTY predictions")
            st.pyplot(fig2)

    else:
        st.info("Click **Predict** to run the models.")








# ---------- BankNIFTY Constituents ----------
with tabs[6]:
    st.header("🏦 BankNIFTY Constituents Predictions & Correlation")

    model_choice = st.selectbox("Model", ["LR","RF","XGB"], index=1, key="bank_model")

    if st.button("🔮 Predict BankNIFTY"):
        with st.spinner("Running predictions... please wait"):
            # constituents
            df_preds = batch_predict(BANKNIFTY, selected_interval, model_choice, forecast_periods)
            # index ETF (BANKBEES trades, so volume features work)
            df_bank = batch_predict(["BANKBEES.NS"], selected_interval, model_choice, forecast_periods)

        if df_preds.empty or df_bank.empty:
            st.warning("Not enough data to generate predictions.")
        else:
            # --- Predictions table ---
            st.subheader("Predicted Closes (last steps + next forecast)")
            df_preds_with_index = df_preds.copy()
            df_preds_with_index["BANKNIFTY"] = df_bank.iloc[:, 0]

            st.dataframe(
                df_preds_with_index.tail().T.reset_index().rename(columns={"index": "Ticker"})
            )

            # --- Pairwise correlation ---
            st.subheader("Correlation Heatmap (full predictions)")
            fig, ax = plt.subplots(figsize=(10,7))
            sns.heatmap(df_preds_with_index.corr(), cmap="RdBu_r", center=0,
                        vmin=-1, vmax=1, annot=False, ax=ax)
            st.pyplot(fig)

            # --- Correlation with BankNIFTY only ---
            st.subheader("Correlation with BankNIFTY predictions")
            corr_with_bank = df_preds_with_index.corr()["BANKNIFTY"].drop("BANKNIFTY")
            corr_sorted = corr_with_bank.sort_values(ascending=False)

            fig2, ax2 = plt.subplots(figsize=(10,6))
            corr_sorted.plot(kind="bar", ax=ax2, color="teal")
            ax2.set_ylabel("Correlation")
            ax2.set_title("Ticker correlation with BankNIFTY predictions")
            st.pyplot(fig2)
    else:
        st.info("Click **Predict BankNIFTY** to run the models.")



