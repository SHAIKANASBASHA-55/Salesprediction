"""
SalesIQ v4 — AI Sales Forecasting + yfinance Real-Time Data
Backend: Pure Python HTTP server
Models:  Random Forest + Gradient Boosting + Deep MLP Ensemble
Data:    yfinance API — real company revenue/stock data (no API key needed!)
Chat:    Built-in rule-based chatbot — zero dependencies, zero API keys
"""

import io, json, math, warnings, traceback, time, csv
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

PORT = 8000
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  YFINANCE DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────

# Popular tickers the user can choose from
TICKER_PRESETS = {
    "Amazon":    "AMZN",
    "Apple":     "AAPL",
    "Microsoft": "MSFT",
    "Tesla":     "TSLA",
    "Google":    "GOOGL",
    "Meta":      "META",
    "Netflix":   "NFLX",
    "Nvidia":    "NVDA",
    "Walmart":   "WMT",
    "Nike":      "NKE",
}

def fetch_yfinance_data(ticker: str, period: str = "2y", interval: str = "1mo"):
    """
    Fetch stock/financial data using yfinance.
    Returns a CSV string ready to save and analyze.
    period options: 1y, 2y, 5y, max
    interval options: 1d, 1wk, 1mo
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError(
            "yfinance not installed. Run: pip install yfinance"
        )

    ticker = ticker.upper().strip()
    tk = yf.Ticker(ticker)

    # Fetch price history
    hist = tk.history(period=period, interval=interval)
    if hist.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    hist = hist.reset_index()
    # Rename for clarity
    hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
    date_col = "date" if "date" in hist.columns else hist.columns[0]
    hist[date_col] = pd.to_datetime(hist[date_col]).dt.strftime("%Y-%m-%d")

    # Add derived features useful for forecasting
    hist["volume_m"]      = (hist["volume"] / 1e6).round(2)
    hist["price_range"]   = (hist["high"] - hist["low"]).round(2)
    hist["month"]         = pd.to_datetime(hist[date_col]).dt.month
    hist["quarter"]       = pd.to_datetime(hist[date_col]).dt.quarter

    # Keep useful columns
    keep = [date_col, "close", "open", "high", "low", "volume_m", "price_range", "month", "quarter"]
    keep = [c for c in keep if c in hist.columns]
    hist = hist[keep].dropna()

    # Save to uploads dir
    filename = f"{ticker}_{period}_{interval}.csv"
    path = UPLOAD_DIR / filename
    hist.to_csv(path, index=False)

    return {
        "filename":  filename,
        "ticker":    ticker,
        "rows":      len(hist),
        "columns":   list(hist.columns),
        "period":    period,
        "interval":  interval,
        "preview":   hist.head(6).fillna("").astype(str).to_dict(orient="records"),
        "dtypes":    {col: str(hist[col].dtype) for col in hist.columns},
        "date_range": f"{hist[date_col].iloc[0]} → {hist[date_col].iloc[-1]}",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  BUILT-IN CHATBOT
# ─────────────────────────────────────────────────────────────────────────────

def handle_chat(message, context_data):
    msg = message.lower().strip()
    d = context_data or {}

    target   = d.get("target", "sales")
    n        = d.get("n", 0)
    mean_val = d.get("mean", 0)
    median   = d.get("median", 0)
    std_val  = d.get("std", 0)
    min_val  = d.get("min", 0)
    max_val  = d.get("max", 0)
    growth   = d.get("growth_pct", 0)
    r2       = d.get("r2", 0)
    rmse     = d.get("rmse", 0)
    mape     = d.get("mape", 0)
    cv_r2    = d.get("cv_r2", 0)
    cv_std   = d.get("cv_std", 0)
    w_rf     = d.get("w_rf", 33)
    w_gbm    = d.get("w_gbm", 33)
    w_mlp    = d.get("w_mlp", 33)
    shap     = d.get("shap", [])
    forecast = d.get("forecast", [])
    skew     = d.get("skew", 0)
    total    = d.get("total", 0)
    features = d.get("features", [])
    ticker   = d.get("ticker", "")

    top1     = shap[0]["feature"] if shap else "N/A"
    top2     = shap[1]["feature"] if len(shap) > 1 else "N/A"
    acc_word = "Excellent" if r2 > 0.85 else "Good" if r2 > 0.7 else "Moderate" if r2 > 0.55 else "Low"
    trend    = "upward" if forecast and len(forecast) > 1 and forecast[-1] > forecast[0] else "downward"

    def fmt(v):
        v = float(v)
        if abs(v) >= 1e6: return f"{v/1e6:.2f}M"
        if abs(v) >= 1e3: return f"{v/1e3:.1f}K"
        return f"{v:.2f}"

    # yfinance specific questions
    if any(w in msg for w in ["yfinance","yahoo","real time","live data","stock","ticker","fetch","api data","company"]):
        tick_str = f" (currently loaded: **{ticker}**)" if ticker else ""
        return (f"**yfinance Integration{tick_str}**\n\n"
                f"SalesIQ uses **yfinance** to pull real financial data — no API key needed!\n\n"
                f"**Available data:**\n"
                f"• Monthly/weekly/daily closing prices\n"
                f"• Volume, price range (high-low)\n"
                f"• Month & quarter features for seasonality\n\n"
                f"**Supported companies:** Amazon, Apple, Microsoft, Tesla, Google, Meta, Netflix, Nvidia, Walmart, Nike — or type any stock ticker!\n\n"
                f"Use the **'Load Live Data'** button to fetch fresh data anytime.")

    if any(w in msg for w in ["hi","hello","hey","hlo","hii","howdy","sup","yo","greet"]):
        tick_part = f" Loaded: **{ticker}**." if ticker else ""
        return f"Hey! I'm your SalesIQ assistant.{tick_part} I've analyzed **{target}** ({n} records) and I'm ready!\n\nTry: 'what drives sales', 'show forecast', 'how accurate', or 'help'!"

    if any(w in msg for w in ["help","what can you","commands","options","menu"]):
        return ("Here's what I can answer:\n\n"
                "📡 **Data** — 'yfinance data', 'live stock data', 'which ticker'\n"
                "📊 **Stats** — 'show stats', 'average', 'total', 'min max'\n"
                "🤖 **Model** — 'how accurate', 'model score', 'explain R2', 'MAPE'\n"
                "🔍 **Features** — 'what drives sales', 'top features', 'SHAP importance'\n"
                "🔭 **Forecast** — 'show forecast', 'next period', 'future values'\n"
                "📐 **Confidence** — 'confidence interval', 'uncertainty'\n"
                "⚙️ **Models** — 'random forest', 'gradient boosting', 'neural network'\n"
                "⚠️ **Risks** — 'any risks', 'limitations', 'warnings'\n"
                "💡 **Tips** — 'how to improve', 'recommendations'\n"
                "📖 **Learn** — 'what is MAPE', 'explain SHAP', 'what is R2'")

    if any(w in msg for w in ["accura","r2","r²","score","reliable","perform","how good","mape","rmse","mae","error rate"]):
        status = "✅ Great" if r2 > 0.75 else "⚠️ Moderate" if r2 > 0.55 else "❌ Low"
        return (f"**{target} Model Performance:**\n\n"
                f"• R² Score: **{r2:.4f}** ({acc_word}) {status}\n"
                f"• MAPE: **{mape:.1f}%** average forecast error\n"
                f"• RMSE: **{fmt(rmse)}**\n"
                f"• CV R²: **{cv_r2}** ± {cv_std} (stability)\n\n"
                f"{'The model is reliable for forecasting.' if r2>0.75 else 'Consider adding more features or data.'}")

    if any(w in msg for w in ["drive","feature","shap","import","factor","influ","top","key","affect","impact","cause"]):
        if not shap:
            return "No feature data yet — run the analysis first!"
        lines = "\n".join([f"{i+1}. **{s['feature']}** — {s['importance']:.1f}%" for i,s in enumerate(shap[:6])])
        return (f"**Top Drivers of {target}:**\n\n{lines}\n\n"
                f"**{top1}** is your #1 lever. **{top2}** is secondary.")

    if any(w in msg for w in ["forecast","predict","future","next","upcoming","project","period","coming"]):
        if not forecast:
            return "No forecast yet — run the analysis first!"
        lines = "\n".join([f"• Period +{i+1}: **{fmt(v)}**" for i,v in enumerate(forecast)])
        delta = ((forecast[-1]-forecast[0])/(abs(forecast[0])+1e-8)*100)
        emoji = "📈" if trend=="upward" else "📉"
        return (f"**{target} Forecast ({len(forecast)} periods):**\n\n{lines}\n\n"
                f"{emoji} Trend: **{trend}** | Change: **{delta:+.1f}%**")

    if any(w in msg for w in ["stat","average","mean","median","total","min","max","overview","summary","number","data","size"]):
        cv_pct = std_val/(mean_val+1e-8)*100
        return (f"**{target} Summary ({n} records):**\n\n"
                f"• Total: **{fmt(total)}**\n"
                f"• Mean: **{fmt(mean_val)}** | Median: **{fmt(median)}**\n"
                f"• Range: **{fmt(min_val)}** → **{fmt(max_val)}**\n"
                f"• Growth: **{growth:+.1f}%** (first→last)\n"
                f"• Skew: {skew:.2f} ({'right' if skew>0.5 else 'left' if skew<-0.5 else 'symmetric'})")

    if any(w in msg for w in ["growth","trend","increas","decreas","up","down"]):
        emoji = "📈" if growth > 0 else "📉"
        return (f"{emoji} **{target}** changed **{growth:+.1f}%** (first → last).\n"
                f"Forecast direction: **{trend}** {emoji}")

    if any(w in msg for w in ["random forest","rf","forest"]):
        return (f"**Random Forest — {w_rf:.0f}% ensemble weight**\n\n"
                f"Trains many decision trees on random subsets, averages predictions.\n"
                f"Strengths: Resistant to overfitting, handles non-linear patterns, provides CI via tree variance.")

    if any(w in msg for w in ["gradient","gbm","boost"]):
        return (f"**Gradient Boosting — {w_gbm:.0f}% ensemble weight**\n\n"
                f"Builds trees sequentially — each corrects errors of the previous.\n"
                f"Strengths: Often most accurate on tabular data.")

    if any(w in msg for w in ["mlp","neural","deep","network"]):
        return (f"**Deep MLP Neural Net — {w_mlp:.0f}% ensemble weight**\n\n"
                f"Architecture: 256→128→64→32, ReLU, Adam optimizer.\n"
                f"Strengths: Learns abstract feature combinations.")

    if any(w in msg for w in ["ensemble","methodology","combin","all model"]):
        best = max([("Random Forest",w_rf),("Gradient Boosting",w_gbm),("MLP Neural Net",w_mlp)],key=lambda x:x[1])
        return (f"**3-Model Ensemble:**\n\n"
                f"• 🌲 Random Forest: **{w_rf:.0f}%**\n"
                f"• 🚀 Gradient Boosting: **{w_gbm:.0f}%**\n"
                f"• 🧠 Deep MLP: **{w_mlp:.0f}%**\n\n"
                f"🏆 Strongest: **{best[0]}** ({best[1]:.0f}%)")

    if "mape" in msg and any(w in msg for w in ["what","explain","mean","define","is"]):
        grade = "Excellent ✅" if mape<10 else "Good 👍" if mape<20 else "Acceptable ⚠️" if mape<30 else "Needs work ❌"
        return (f"**MAPE = Mean Absolute Percentage Error**\n\n"
                f"Average % gap between predicted vs actual.\n\n"
                f"Your MAPE: **{mape:.1f}%** → {grade}")

    if any(w in msg for w in ["what is r2","explain r2","r squared"]):
        return (f"**R² = Coefficient of Determination**\n\n"
                f"How much variance in {target} the model explains.\n\n"
                f"Your R²: **{r2:.4f}** — explains **{r2*100:.1f}%** of variance → **{acc_word}**")

    if any(w in msg for w in ["risk","concern","problem","warn","limit","weakness"]):
        risks = ["• Forecast assumes linear trends — won't capture sudden market shifts"]
        if cv_std > 0.1: risks.append(f"• CV std {cv_std} is high — model varies across periods")
        if mape > 20: risks.append(f"• MAPE {mape:.1f}% — consider more features")
        if n < 50: risks.append(f"• Only {n} records — more data improves reliability")
        return "**Risk Factors:**\n\n" + "\n".join(risks)

    if any(w in msg for w in ["improve","recommend","better","tip","suggestion","advice"]):
        return (f"**Recommendations for {target}:**\n\n"
                f"1. 🎯 Focus on **{top1}** — top driver\n"
                f"2. 📅 Retrain monthly with fresh yfinance data\n"
                f"3. 📊 Try different tickers for comparison\n"
                f"4. ⏱ Use longer periods (5y) for more training data\n"
                f"5. 📐 Use CI bands for scenario planning")

    if any(w in msg for w in ["thank","thanks","great","awesome","nice","good job","perfect","cool"]):
        return f"You're welcome! Happy to help with **{target}** analysis. Ask me anything!"

    return (f"I specialize in your **{target}** dataset. Didn't quite get that.\n\n"
            f"Try:\n• 'yfinance data' — about live data\n"
            f"• 'what drives {target}'\n• 'how accurate is the model'\n"
            f"• 'show forecast'\n• 'help' — full list")


# ─────────────────────────────────────────────────────────────────────────────
#  ML PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df, feature_cols, target_col):
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    for col in feature_cols:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    return df


def hyperparameter_search(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=min(4, len(y_train) // 3))
    rf_configs = [
        dict(n_estimators=100, max_depth=None, min_samples_split=2),
        dict(n_estimators=200, max_depth=10,   min_samples_split=2),
        dict(n_estimators=150, max_depth=None, min_samples_split=4),
    ]
    best_rf_cfg, best_rf_score = rf_configs[0], -np.inf
    for cfg in rf_configs:
        scores = []
        for tr, val in tscv.split(X_train):
            m = RandomForestRegressor(**cfg, random_state=42, n_jobs=-1)
            m.fit(X_train[tr], y_train[tr])
            scores.append(r2_score(y_train[val], m.predict(X_train[val])))
        s = np.mean(scores)
        if s > best_rf_score:
            best_rf_score = s; best_rf_cfg = cfg

    gbm_configs = [
        dict(n_estimators=100, learning_rate=0.1,  max_depth=4),
        dict(n_estimators=150, learning_rate=0.05, max_depth=5),
        dict(n_estimators=200, learning_rate=0.08, max_depth=3),
    ]
    best_gbm_cfg, best_gbm_score = gbm_configs[0], -np.inf
    for cfg in gbm_configs:
        scores = []
        for tr, val in tscv.split(X_train):
            m = GradientBoostingRegressor(**cfg, random_state=42)
            m.fit(X_train[tr], y_train[tr])
            scores.append(r2_score(y_train[val], m.predict(X_train[val])))
        s = np.mean(scores)
        if s > best_gbm_score:
            best_gbm_score = s; best_gbm_cfg = cfg

    return best_rf_cfg, best_gbm_cfg, best_rf_score, best_gbm_score


def compute_confidence_intervals(rf_model, X, percentile=90):
    tree_preds = np.array([tree.predict(X) for tree in rf_model.estimators_])
    lower = np.percentile(tree_preds, (100-percentile)/2, axis=0)
    upper = np.percentile(tree_preds, 100-(100-percentile)/2, axis=0)
    return lower.tolist(), upper.tolist()


def shap_permutation(model, X, y, feature_names, scaler=None, n_repeats=15):
    X_in = scaler.transform(X) if scaler else X
    baseline = r2_score(y, model.predict(X_in))
    importances = []
    rng = np.random.RandomState(42)
    for i in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_in.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            drops.append(baseline - r2_score(y, model.predict(X_perm)))
        importances.append(np.mean(drops))
    imp = np.clip(np.array(importances), 0, None)
    imp = (imp / (imp.sum() + 1e-9)) * 100
    return [{"feature": feature_names[i], "importance": round(float(imp[i]), 2)} for i in range(len(feature_names))]


def run_full_analysis(filename, target_col, feature_cols, forecast_periods=6):
    t0 = time.time()
    df = pd.read_csv(UPLOAD_DIR / filename)
    df = clean_data(df, feature_cols, target_col)
    if len(df) < 6:
        raise ValueError("Need at least 6 valid rows after cleaning.")

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    n = len(y)

    test_size = max(2, int(n * 0.2))
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    rf_cfg, gbm_cfg, _, _ = hyperparameter_search(X_train, y_train)

    rf = RandomForestRegressor(**rf_cfg, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_test = rf.predict(X_test)
    rf_r2 = max(0.001, r2_score(y_test, rf_test))

    gbm = GradientBoostingRegressor(**gbm_cfg, random_state=42)
    gbm.fit(X_train, y_train)
    gbm_test = gbm.predict(X_test)
    gbm_r2 = max(0.001, r2_score(y_test, gbm_test))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32), activation="relu", solver="adam",
        max_iter=2000, random_state=42, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=25,
        learning_rate_init=0.001, batch_size="auto", alpha=0.0001,
    )
    mlp.fit(X_train_s, y_train)
    mlp_test = mlp.predict(X_test_s)
    mlp_r2 = max(0.001, r2_score(y_test, mlp_test))

    total_r2 = rf_r2 + gbm_r2 + mlp_r2
    w_rf = rf_r2/total_r2; w_gbm = gbm_r2/total_r2; w_mlp = mlp_r2/total_r2
    ens_test = w_rf*rf_test + w_gbm*gbm_test + w_mlp*mlp_test

    r2   = r2_score(y_test, ens_test)
    rmse = math.sqrt(mean_squared_error(y_test, ens_test))
    mae  = mean_absolute_error(y_test, ens_test)
    mape = float(np.mean(np.abs((y_test-ens_test)/(np.abs(y_test)+1e-8)))*100)

    tscv = TimeSeriesSplit(n_splits=min(5, n//3))
    cv_scores = []
    for tr, val in tscv.split(X):
        rf_tmp = RandomForestRegressor(**rf_cfg, random_state=42, n_jobs=-1)
        rf_tmp.fit(X[tr], y[tr])
        cv_scores.append(r2_score(y[val], rf_tmp.predict(X[val])))
    cv_mean = round(float(np.mean(cv_scores)), 4)
    cv_std  = round(float(np.std(cv_scores)),  4)

    X_all_s = scaler.transform(X)
    rf_all  = rf.predict(X)
    gbm_all = gbm.predict(X)
    mlp_all = mlp.predict(X_all_s)
    pred_all = (w_rf*rf_all + w_gbm*gbm_all + w_mlp*mlp_all).tolist()

    ci_lo, ci_hi = compute_confidence_intervals(rf, X, percentile=90)

    shap_rf  = shap_permutation(rf,  X_test, y_test, feature_cols)
    shap_mlp = shap_permutation(mlp, X_test, y_test, feature_cols, scaler=scaler)
    shap_combined = []
    for i, col in enumerate(feature_cols):
        avg = (shap_rf[i]["importance"] + shap_mlp[i]["importance"]) / 2
        shap_combined.append({"feature": col, "importance": round(avg, 2)})
    shap_combined.sort(key=lambda x: x["importance"], reverse=True)
    total_imp = sum(s["importance"] for s in shap_combined) + 1e-9
    for s in shap_combined:
        s["importance"] = round(s["importance"]/total_imp*100, 2)

    last_X = X[-1].copy()
    trends = (X[-1]-X[0]) / max(n-1, 1)
    forecast_vals, fc_lo, fc_hi = [], [], []
    cur_X = last_X.copy()
    for step in range(forecast_periods):
        cur_X = cur_X + trends
        fc = (w_rf*rf.predict(cur_X.reshape(1,-1))[0] +
              w_gbm*gbm.predict(cur_X.reshape(1,-1))[0] +
              w_mlp*mlp.predict(scaler.transform(cur_X.reshape(1,-1)))[0])
        spread = rmse*(1+step*0.15)
        forecast_vals.append(round(float(fc), 2))
        fc_lo.append(round(float(fc-spread), 2))
        fc_hi.append(round(float(fc+spread), 2))

    growth_pct = float((y[-1]-y[0])/(abs(y[0])+1e-8)*100)
    stats = {
        "n": n, "mean": round(float(y.mean()),2),
        "median": round(float(np.median(y)),2),
        "std": round(float(y.std()),2),
        "min": round(float(y.min()),2), "max": round(float(y.max()),2),
        "total": round(float(y.sum()),2),
        "growth_pct": round(growth_pct,2),
        "skew": round(float(pd.Series(y).skew()),3),
    }

    counts, edges = np.histogram(y, bins=12)
    distribution = {
        "labels": [f"{edges[i]:.0f}" for i in range(len(edges)-1)],
        "counts": counts.tolist(),
    }
    pop = [round(float((y[i]-y[i-1])/(abs(y[i-1])+1e-8)*100),2) for i in range(1,n)]

    elapsed = round(time.time()-t0, 2)
    return {
        "status":"success","elapsed":elapsed,
        "stats":stats,
        "metrics":{
            "r2":round(r2,4),"rmse":round(rmse,2),
            "mae":round(mae,2),"mape":round(mape,2),
            "cv_r2":cv_mean,"cv_std":cv_std,
            "w_rf":round(w_rf*100,1),"w_gbm":round(w_gbm*100,1),"w_mlp":round(w_mlp*100,1),
            "rf_cfg":rf_cfg,"gbm_cfg":gbm_cfg,
        },
        "shap":shap_combined,
        "historical":[round(float(v),2) for v in y],
        "predicted":[round(v,2) for v in pred_all],
        "rf_pred":[round(float(v),2) for v in rf_all],
        "gbm_pred":[round(float(v),2) for v in gbm_all],
        "mlp_pred":[round(float(v),2) for v in mlp_all],
        "ci_lo":ci_lo,"ci_hi":ci_hi,
        "forecast":forecast_vals,"fc_lo":fc_lo,"fc_hi":fc_hi,
        "pop_growth":pop,"distribution":distribution,
        "report":"",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP SERVER
# ─────────────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  {args[0]} {args[1]}")

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",len(body))
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, mime):
        try:
            with open(path,"rb") as f: data = f.read()
            self.send_response(200)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",len(data))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_response(404); self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/","/index.html"):
            self.send_file("static/index.html","text/html")
        else:
            mime = {".css":"text/css",".js":"application/javascript"}.get(Path(path).suffix,"text/plain")
            self.send_file("static"+path, mime)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length",0))
        body = self.rfile.read(length)
        if   path == "/upload":       self._handle_upload(body)
        elif path == "/fetch_live":   self._handle_fetch_live(body)
        elif path == "/analyze":      self._handle_analyze(body)
        elif path == "/chat":         self._handle_chat(body)
        elif path == "/tickers":      self.send_json(TICKER_PRESETS)
        else: self.send_json({"error":"Not found"},404)

    def _handle_upload(self, body):
        try:
            content_type = self.headers.get("Content-Type","")
            boundary = None
            for part in content_type.split(";"):
                p = part.strip()
                if p.startswith("boundary="):
                    boundary = p[9:].strip('"'); break
            if not boundary:
                self.send_json({"error":"No boundary"},400); return
            boundary_bytes = ("--"+boundary).encode()
            parts = body.split(boundary_bytes)
            filename = "upload.csv"; file_data = None
            for part in parts:
                if b"filename=" in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end == -1: continue
                    headers_raw = part[:header_end].decode("utf-8",errors="ignore")
                    for h in headers_raw.split("\r\n"):
                        if "filename=" in h:
                            fname = h.split("filename=")[-1].strip().strip('"')
                            if fname: filename = fname
                    file_data = part[header_end+4:].rstrip(b"\r\n--")
            if file_data is None:
                self.send_json({"error":"No file found"},400); return
            with open(UPLOAD_DIR/filename,"wb") as f: f.write(file_data)
            df = pd.read_csv(io.BytesIO(file_data))
            preview = df.head(6).fillna("").astype(str).to_dict(orient="records")
            dtypes = {col:str(df[col].dtype) for col in df.columns}
            self.send_json({"filename":filename,"rows":len(df),"columns":list(df.columns),"dtypes":dtypes,"preview":preview})
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error":str(e)},500)

    def _handle_fetch_live(self, body):
        """Fetch live data from yfinance and return as upload-compatible response."""
        try:
            req = json.loads(body)
            ticker   = req.get("ticker", "AMZN")
            period   = req.get("period", "2y")
            interval = req.get("interval", "1mo")
            result = fetch_yfinance_data(ticker, period, interval)
            self.send_json(result)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": str(e)}, 500)

    def _handle_analyze(self, body):
        try:
            req = json.loads(body)
            result = run_full_analysis(
                filename=req["filename"],target_col=req["target_col"],
                feature_cols=req["feature_cols"],
                forecast_periods=req.get("forecast_periods",6),
            )
            self.send_json(result)
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error":str(e)},500)

    def _handle_chat(self, body):
        try:
            req = json.loads(body)
            message = req.get("message","").strip()
            context_data = req.get("context_data",{})
            if not message:
                self.send_json({"reply":""}); return
            reply = handle_chat(message, context_data)
            self.send_json({"reply":reply})
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error":str(e),"reply":f"Error: {str(e)}"},500)


if __name__ == "__main__":
    # Check yfinance availability at startup
    try:
        import yfinance
        yf_status = f"yfinance {yfinance.__version__} ✓"
    except ImportError:
        yf_status = "yfinance NOT installed — run: pip install yfinance"

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  SalesIQ v4 — Real-Time Data + No API Key!              ║
║  ML: RF + GBM + Deep MLP Ensemble                       ║
║  Data: {yf_status:<44} ║
╠══════════════════════════════════════════════════════════╣
║  → http://localhost:8000                                ║
╚══════════════════════════════════════════════════════════╝
""")
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")