"""
Machine learning signal filter.

Trains a GradientBoostingClassifier on historical trade outcomes
(from backtest checkpoints) to predict win probability at entry.

Workflow:
  1. Run backtest once to generate labeled trades with entry indicators
  2. python ml_signal.py --train     -- trains + saves model
  3. python ml_signal.py --report    -- shows accuracy + feature importance
  4. Subsequent backtests/live signals automatically use the model

The model is intentionally conservative (threshold 0.52) to avoid
over-filtering — it only skips trades it's clearly low-confidence about.
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")

console = Console()

MODEL_PATH         = Path("ml_model.pkl")
CHECKPOINT_DIR     = Path("backtest_checkpoints")
WIN_PROB_THRESHOLD = 0.52   # only skip trades model thinks have < 52% win chance

# Feature vector definition — must stay consistent between training and inference
FEATURE_NAMES = [
    "rsi",           # RSI(14) at entry — momentum
    "macd_hist",     # MACD histogram at entry — trend momentum
    "bb_pct_b",      # Bollinger %B at entry — mean reversion position
    "ema_ratio",     # EMA20 / EMA50 — trend alignment
    "hv20",          # Historical volatility — cost of premium
    "adx",           # ADX — trend strength
    "vol_ratio",     # Volume vs 20-day avg — conviction
    "atr_pct",       # ATR / stock price — relative volatility
    "direction_enc", # 1.0 = CALL, -1.0 = PUT
]


# ── Feature extraction ─────────────────────────────────────────────────────────

def trade_to_features(trade) -> dict | None:
    """Extract feature dict from a Trade that has entry_indicators set."""
    ind = getattr(trade, "entry_indicators", None)
    if not ind:
        return None
    try:
        ema20 = ind.get("ema20", 1.0) or 1.0
        ema50 = ind.get("ema50", 1.0) or 1.0
        return {
            "rsi":           float(ind.get("rsi",       50.0)),
            "macd_hist":     float(ind.get("macd_hist",  0.0)),
            "bb_pct_b":      float(ind.get("bb_pct_b",   0.5)),
            "ema_ratio":     float(ema20) / max(float(ema50), 1e-6),
            "hv20":          float(ind.get("hv20",       0.3)),
            "adx":           float(ind.get("adx",       20.0)),
            "vol_ratio":     float(ind.get("vol_ratio",  1.0)),
            "atr_pct":       float(ind.get("atr_pct",   0.02)),
            "direction_enc": 1.0 if trade.direction == "CALL" else -1.0,
        }
    except Exception:
        return None


def features_to_row(feat: dict) -> list[float]:
    return [feat.get(k, 0.0) for k in FEATURE_NAMES]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load all completed trades from checkpoints → (X features, y labels)."""
    X, y = [], []

    for path in CHECKPOINT_DIR.glob("*.pkl"):
        try:
            with open(path, "rb") as f:
                ck = pickle.load(f)
        except Exception:
            continue

        for trade in ck.get("trades", []):
            if trade.exit_date is None or trade.pnl_dollars is None:
                continue
            feat = trade_to_features(trade)
            if feat is None:
                continue
            row = features_to_row(feat)
            if any(np.isnan(v) or np.isinf(v) for v in row):
                continue
            X.append(row)
            y.append(1 if trade.is_win else 0)

    return np.array(X, dtype=float), np.array(y, dtype=int)


# ── Model training ─────────────────────────────────────────────────────────────

def train_model() -> Pipeline | None:
    """Train GradientBoosting on historical trades. Returns fitted pipeline."""
    console.print("\n[cyan]Loading training data from checkpoints...[/cyan]")
    X, y = load_training_data()

    if len(X) < 100:
        console.print(
            f"[red]Not enough training data ({len(X)} trades with entry indicators).[/red]\n"
            "[yellow]Run the backtest first, then train: python ml_signal.py --train[/yellow]"
        )
        return None

    n_wins   = int(y.sum())
    n_losses = len(y) - n_wins
    console.print(
        f"[green]Training on {len(X)} trades[/green] "
        f"({n_wins} wins / {n_losses} losses, base rate {n_wins/len(X)*100:.1f}%)"
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators    = 300,
            max_depth       = 3,        # shallow trees reduce overfitting
            learning_rate   = 0.04,
            subsample       = 0.75,     # stochastic gradient boosting
            min_samples_leaf= 20,       # prevent fitting to tiny subsets
            random_state    = 42,
        )),
    ])

    # 5-fold stratified CV for honest accuracy estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc  = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    cv_acc  = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    console.print(f"\n[bold]Cross-Validation Results (5-fold):[/bold]")
    console.print(f"  ROC-AUC : {cv_roc.mean():.3f} +/- {cv_roc.std():.3f}")
    console.print(f"  Accuracy: {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")
    console.print(
        f"\n  [dim]Note: base rate (random) accuracy = {n_wins/len(X)*100:.1f}%. "
        f"Model adds {(cv_acc.mean() - n_wins/len(X))*100:.1f}pp lift.[/dim]"
    )

    # Train on full dataset
    pipeline.fit(X, y)

    # Feature importance table
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_

    table = Table(title="Feature Importance", box=box.SIMPLE_HEAVY, border_style="blue")
    table.add_column("Feature",     style="cyan", width=16)
    table.add_column("Importance",  justify="right", width=12)
    table.add_column("Bar",         width=30)

    max_imp = max(importances)
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp / max_imp * 25)
        table.add_row(name, f"{imp:.4f}", f"[green]{bar}[/green]")
    console.print(table)

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    console.print(f"\n[green]Model saved to {MODEL_PATH}[/green]")
    console.print(
        f"[dim]Threshold: {WIN_PROB_THRESHOLD:.0%} — trades below this predicted "
        f"win probability will be skipped.[/dim]\n"
    )

    return pipeline


# ── Inference ──────────────────────────────────────────────────────────────────

def load_model() -> Pipeline | None:
    """Load saved model. Returns None if no model exists yet."""
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def predict_win_prob(model: Pipeline, features: dict) -> float:
    """Predict win probability (0.0 to 1.0) for a signal."""
    if model is None:
        return 0.5
    try:
        row = [features_to_row(features)]
        if any(np.isnan(v) or np.isinf(v) for v in row[0]):
            return 0.5
        return float(model.predict_proba(row)[0][1])
    except Exception:
        return 0.5


def ml_filter(
    model:     Pipeline | None,
    features:  dict,
    threshold: float = WIN_PROB_THRESHOLD,
) -> tuple[bool, str]:
    """Filter gate: skip if predicted win probability below threshold."""
    if model is None:
        return True, ""
    prob = predict_win_prob(model, features)
    if prob < threshold:
        return False, f"ML model: {prob:.1%} win probability (threshold {threshold:.0%})"
    return True, ""


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML signal filter — options trading bot")
    parser.add_argument("--train",  action="store_true", help="Train model from backtest checkpoints")
    parser.add_argument("--report", action="store_true", help="Show model accuracy report")
    args = parser.parse_args()

    if args.train or args.report:
        train_model()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
