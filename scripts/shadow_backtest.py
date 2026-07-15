"""
Shadow backtest — replica a logica exata do runner.py de producao (equity,
fees, stop-loss, cooldown) sobre dados historicos, para testar hipoteses de
correcao SEM arriscar producao.

Compara contra o log.csv real (mai-jul/2026) para validar que a simulacao
bate com o que aconteceu ao vivo, depois testa overlays candidatos.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import ta
from stable_baselines3 import PPO

from config import ENV, WALK_FORWARD
from features.builder import build_features, FEATURE_COLS
from data.loader import fetch_ohlcv

MODEL_PATH = Path(WALK_FORWARD["model_dir"]) / "wf_window_4.zip"
WINDOW = ENV["window_size"]
INITIAL_CAP = ENV["initial_capital"]
FEE = ENV["fee_rate"] + ENV["slippage_rate"]

ACTION_TO_POS = {0: 0, 1: 1, 2: -1}
ACTION_MAP = {0: "FLAT", 1: "LONG", 2: "SHORT"}


def simulate(
    df: pd.DataFrame,
    model: PPO,
    start_idx: int,
    end_idx: int,
    stop_loss_pct: float | None = 0.04,
    cooldown_bars: int = 3,
    chop_adx_threshold: float | None = None,
    min_hold_bars: int = 0,
    adx: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Replica o loop candle-a-candle do runner.py.

    Parametros do overlay (todos testaveis):
      stop_loss_pct:      None desativa. Corta posicao se unrealized < -X.
      cooldown_bars:       candles em FLAT forcado apos um STOP.
      chop_adx_threshold:  None desativa. Se ADX(14) < threshold, bloqueia
                            ABERTURA de nova posicao direcional (chop filter).
      min_hold_bars:       0 desativa. Minimo de candles antes de permitir
                            troca voluntaria de posicao (nao afeta STOP).
    """
    feats  = df[FEATURE_COLS].values.astype("float32")
    closes = df["close"].values.astype(float)
    times  = df.index

    position     = 0
    equity       = INITIAL_CAP
    entry_price  = 0.0
    steps_in_pos = 0
    peak_equity  = INITIAL_CAP
    n_trades     = 0
    cooldown_left = 0
    recent_was_stop = False

    records = []

    for i in range(start_idx, end_idx):
        price = closes[i]
        prev  = closes[i - 1]
        ret   = (price - prev) / prev if prev > 0 else 0.0

        equity *= (1.0 + position * ret)

        obs_window = np.clip(feats[i - WINDOW + 1:i + 1].flatten(), -5.0, 5.0)
        unrealized = (
            position * (price - entry_price) / entry_price
            if entry_price > 0 and position != 0 else 0.0
        )
        drawdown  = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        time_norm = min(steps_in_pos / WINDOW, 1.0)
        obs = np.concatenate([
            obs_window,
            np.array([float(position), unrealized, drawdown, time_norm], dtype="float32"),
        ])

        action, _ = model.predict(obs, deterministic=True)
        model_pos = ACTION_TO_POS[int(action)]
        new_position = model_pos
        decision = ACTION_MAP[int(action)]

        # --- Overlay 1: cooldown apos stop ---
        if cooldown_left > 0:
            new_position = 0
            decision = "COOLDOWN"
            cooldown_left -= 1

        # --- Overlay 2: stop-loss por posicao ---
        elif stop_loss_pct is not None and position != 0 and unrealized < -stop_loss_pct:
            new_position = 0
            decision = "STOP"
            cooldown_left = cooldown_bars

        # --- Overlay 3: chop filter (bloqueia ABERTURA em mercado sem trend) ---
        elif (chop_adx_threshold is not None and adx is not None
              and new_position != position and new_position != 0
              and adx.iloc[i] < chop_adx_threshold):
            new_position = position if position != 0 else 0
            decision = "CHOP_SKIP" if new_position == 0 else decision

        # --- Overlay 4: holding minimo (nao aplica se for STOP) ---
        elif (min_hold_bars > 0 and position != 0 and new_position != position
              and steps_in_pos < min_hold_bars and decision != "STOP"):
            new_position = position
            decision = "HOLD_MIN"

        if new_position != position:
            n_trans = 1 if (position == 0 or new_position == 0) else 2
            equity *= (1.0 - n_trans * FEE)
            n_trades += 1
            entry_price  = price
            steps_in_pos = 0
        else:
            steps_in_pos += 1

        peak_equity = max(peak_equity, equity)

        records.append({
            "timestamp": times[i], "preco": price, "decisao": decision,
            "posicao_anterior": position, "posicao_atual": new_position,
            "equity": equity, "n_trades": n_trades,
        })
        position = new_position

    return pd.DataFrame(records)


def metrics(sim: pd.DataFrame, label: str) -> dict:
    eq0, eq1 = sim["equity"].iloc[0], sim["equity"].iloc[-1]
    p0, p1   = sim["preco"].iloc[0], sim["preco"].iloc[-1]
    ag  = (eq1 / eq0 - 1) * 100
    bh  = (p1 / p0 - 1) * 100
    peak = sim["equity"].cummax()
    mdd = ((sim["equity"] - peak) / peak * 100).min()
    n_trades = int(sim["n_trades"].iloc[-1])
    n_stop = int((sim["decisao"] == "STOP").sum())
    return {
        "label": label, "retorno_agente": ag, "retorno_bh": bh,
        "alpha": ag - bh, "mdd": mdd, "trades": n_trades, "stops": n_stop,
    }


def main() -> None:
    print("Carregando dados frescos...")
    raw = fetch_ohlcv(exchange_name="binance", symbol="BTC/USDT", timeframe="1h", days=910)
    df = build_features(raw).reset_index(drop=True)
    df.index = raw.index  # preserva timestamps

    # ADX(14) para o chop filter — calculado FORA do vetor de features do modelo
    adx_ind = ta.trend.ADXIndicator(high=raw["high"], low=raw["low"], close=raw["close"], window=14)
    adx = adx_ind.adx().reindex(df.index).fillna(0.0)
    adx.index = range(len(adx))  # alinha com indice posicional do df

    model = PPO.load(str(MODEL_PATH))

    # Janela de teste: periodo real do paper trading (15/mai -> 15/jul)
    start_ts = pd.Timestamp("2026-05-15 21:00", tz="UTC")
    end_ts   = pd.Timestamp("2026-07-15 15:00", tz="UTC")
    start_idx = df.index[df.index >= start_ts][0] if isinstance(df.index, pd.DatetimeIndex) else None

    # df.index foi resetado para RangeIndex; usa raw.index (datetime) para localizar
    raw_idx = raw.index
    start_idx = int(np.searchsorted(raw_idx.values, start_ts.to_datetime64()))
    end_idx   = int(np.searchsorted(raw_idx.values, end_ts.to_datetime64()))
    start_idx = max(start_idx, WINDOW)

    print(f"Periodo de teste: {raw_idx[start_idx]} -> {raw_idx[end_idx-1]}  ({end_idx-start_idx} candles)")
    print()

    configs = [
        dict(label="0. SEM overlay (so modelo puro)", stop_loss_pct=None, cooldown_bars=0),
        dict(label="1. Deployed atual (stop 4% + cooldown 3)", stop_loss_pct=0.04, cooldown_bars=3),
        dict(label="2. Chop filter ADX<20 (sem stop)", stop_loss_pct=None, cooldown_bars=0,
             chop_adx_threshold=20, adx=adx),
        dict(label="3. Chop filter ADX<25 (sem stop)", stop_loss_pct=None, cooldown_bars=0,
             chop_adx_threshold=25, adx=adx),
        dict(label="4. Min hold 6h (sem stop, sem chop)", stop_loss_pct=None, cooldown_bars=0,
             min_hold_bars=6),
        dict(label="5. Min hold 12h (sem stop, sem chop)", stop_loss_pct=None, cooldown_bars=0,
             min_hold_bars=12),
        dict(label="6. Chop ADX<20 + stop 4% + cooldown 3", stop_loss_pct=0.04, cooldown_bars=3,
             chop_adx_threshold=20, adx=adx),
        dict(label="7. Chop ADX<25 + min hold 6h + stop 4%", stop_loss_pct=0.04, cooldown_bars=3,
             chop_adx_threshold=25, adx=adx, min_hold_bars=6),
        dict(label="8. Min hold 6h + stop 4% + cooldown 3", stop_loss_pct=0.04, cooldown_bars=3,
             min_hold_bars=6),
    ]

    results = []
    for cfg in configs:
        label = cfg.pop("label")
        sim = simulate(df, model, start_idx, end_idx, **cfg)
        m = metrics(sim, label)
        results.append(m)
        print(f"{label:45s} | agente {m['retorno_agente']:+7.2f}% | B&H {m['retorno_bh']:+7.2f}% | "
              f"alpha {m['alpha']:+7.2f}pp | MDD {m['mdd']:+7.2f}% | trades {m['trades']:4d} | stops {m['stops']:3d}")

    print()
    print("Referencia (log.csv real de producao, mesmo periodo):")
    print("  agente -24.03% | B&H -17.80% | alpha -6.23pp | MDD -24.97% | trades 138")


if __name__ == "__main__":
    main()
