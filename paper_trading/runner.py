"""
Paper Trading Runner — executa 1x por hora via GitHub Actions.

O que faz a cada execucao:
  1. Busca os ultimos N candles BTC/USDT 1h da Binance (sem API Key — dados publicos)
  2. Constroi as features (identico ao treino — sem look-ahead)
  3. Carrega o modelo V14 (janela 4 — mais dados de treino)
  4. Decide: Long / Short / Flat
  5. Registra decisao + P&L simulado no log CSV
  6. Detecta mudancas de posicao (trades) e reporta

Variaveis de ambiente (GitHub Secrets):
  BINANCE_API_KEY    — opcional, aumenta rate limit (leitura apenas)
  BINANCE_API_SECRET — opcional, par da key acima

Sem API Key tambem funciona — dados OHLCV sao publicos na Binance.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Garante imports do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
from stable_baselines3 import PPO

from config import ENV, WALK_FORWARD
from features.builder import build_features

# ------------------------------------------------------------------
# Configuracao
# ------------------------------------------------------------------

SYMBOL        = "XBTUSD"    # par BTC/USD no Kraken (exchange americana — sem bloqueio geo)
TIMEFRAME     = "1h"
MODEL_PATH    = Path(WALK_FORWARD["model_dir"]) / "wf_window_4.zip"
LOG_PATH      = Path(__file__).parent / "log.csv"
CANDLES_NEEDED = ENV["window_size"] + 200 + 10   # janela obs + warm-up features

POSITION_MAP  = {1: "LONG", 0: "FLAT", -1: "SHORT"}
ACTION_MAP    = {0: "FLAT", 1: "LONG", 2: "SHORT"}
ACTION_TO_POS = {0: 0, 1: 1, 2: -1}

# ------------------------------------------------------------------
# Busca candles ao vivo via Kraken (exchange EUA — sem bloqueio geo)
# ------------------------------------------------------------------

def fetch_live_candles() -> pd.DataFrame:
    """
    Busca os ultimos CANDLES_NEEDED candles 1h do BTC/USD via API publica
    do Kraken. Nao requer autenticacao e funciona de qualquer servidor,
    incluindo GitHub Actions (Azure EUA). Retorna ate 720 candles por request.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        "pair":     SYMBOL,
        "interval": 60,     # minutos
    }

    print(f"[kraken] Buscando candles 1h de {SYMBOL}...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data["error"]:
        raise RuntimeError(f"[kraken] Erro na API: {data['error']}")

    # Nome da chave varia ligeiramente (ex: 'XXBTZUSD')
    pair_key = [k for k in data["result"] if k != "last"][0]
    rows = data["result"][pair_key]

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    # Kraken usa timestamps em SEGUNDOS (nao ms)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(np.int64), unit="s", utc=True
    )
    df[["open", "high", "low", "close", "volume"]] = (
        df[["open", "high", "low", "close", "volume"]].astype(float)
    )
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.iloc[-CANDLES_NEEDED:]   # garante tamanho exato

    print(f"[kraken] {len(df)} candles recebidos. Ultimo: {df.index[-1]}")
    return df


# ------------------------------------------------------------------
# Carrega ou inicializa o log
# ------------------------------------------------------------------

def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return pd.DataFrame(columns=[
        "timestamp", "preco", "decisao", "posicao_anterior",
        "posicao_atual", "equity", "retorno_candle_pct",
        "retorno_acumulado_pct", "n_trades"
    ])


def save_log(df: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


# ------------------------------------------------------------------
# Estado atual da posicao e equity
# ------------------------------------------------------------------

def get_current_state(log: pd.DataFrame) -> tuple[int, float, int]:
    """
    Retorna (posicao_atual, equity_atual, n_trades_total) com base no log.
    Se o log estiver vazio, retorna o estado inicial.
    """
    if log.empty:
        return 0, float(ENV["initial_capital"]), 0

    last = log.iloc[-1]
    return (
        int(last["posicao_atual"]),
        float(last["equity"]),
        int(last["n_trades"]),
    )


# ------------------------------------------------------------------
# Calcula equity apos o candle
# ------------------------------------------------------------------

def update_equity(equity: float, position: int,
                  prev_close: float, curr_close: float) -> tuple[float, float]:
    """
    Atualiza equity com o retorno do candle anterior (o que ja fechou).
    Retorna (nova_equity, retorno_pct).
    """
    if prev_close <= 0:
        return equity, 0.0

    bar_return = (curr_close - prev_close) / prev_close
    new_equity = equity * (1.0 + position * bar_return)
    return new_equity, bar_return * 100.0


def apply_trade_cost(equity: float, old_pos: int, new_pos: int) -> float:
    """Deduz taxa real se houve mudanca de posicao."""
    if old_pos == new_pos:
        return equity
    n_trans = 1 if (old_pos == 0 or new_pos == 0) else 2
    cost = n_trans * (ENV["fee_rate"] + ENV["slippage_rate"])
    return equity * (1.0 - cost)


# ------------------------------------------------------------------
# Runner principal
# ------------------------------------------------------------------

def run() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*55}")
    print(f"  PAPER TRADING — {now}")
    print(f"{'='*55}")

    # 1. Dados ao vivo
    df_raw = fetch_live_candles()

    # 2. Features (identico ao treino)
    df = build_features(df_raw).reset_index(drop=True)

    current_price = float(df["close"].iloc[-1])
    prev_price    = float(df["close"].iloc[-2])

    # 3. Modelo
    if not MODEL_PATH.exists():
        print(f"[ERRO] Modelo nao encontrado: {MODEL_PATH}")
        print("       Certifique-se de que models/wf/wf_window_4.zip esta no repositorio.")
        sys.exit(1)

    model = PPO.load(str(MODEL_PATH))
    print(f"[modelo] V14 carregado de {MODEL_PATH}")

    # 4. Observacao: ultimos window_size candles
    from env.trading_env import TradingEnv
    # Usa os ultimos dados para montar a observacao manualmente
    window = ENV["window_size"]
    from features.builder import FEATURE_COLS
    obs_window = df[FEATURE_COLS].iloc[-window:].values.flatten().astype("float32")
    obs_window = np.clip(obs_window, -5.0, 5.0)

    # Estado de conta simulado (baseado no log)
    log          = load_log()
    old_position, equity, n_trades = get_current_state(log)

    # Unrealized PnL e drawdown para completar obs de conta
    entry_price = 0.0
    if not log.empty and old_position != 0:
        # Busca o preco da ultima entrada
        entries = log[log["posicao_anterior"] == 0]
        if not entries.empty:
            entry_price = float(entries.iloc[-1]["preco"])

    unrealized = (
        old_position * (current_price - entry_price) / entry_price
        if entry_price > 0 and old_position != 0 else 0.0
    )
    peak_equity = float(log["equity"].max()) if not log.empty else equity
    drawdown    = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
    time_norm   = 0.5  # aproximacao conservadora

    account_obs = np.array(
        [float(old_position), unrealized, drawdown, time_norm],
        dtype="float32",
    )
    obs = np.concatenate([obs_window, account_obs])

    # 5. Decisao do agente
    action, _ = model.predict(obs, deterministic=True)
    action     = int(action)
    new_position = ACTION_TO_POS[action]
    decision_str = ACTION_MAP[action]

    # 6. Atualiza equity: primeiro aplica retorno do candle anterior, depois custo
    equity, ret_candle = update_equity(equity, old_position, prev_price, current_price)
    equity = apply_trade_cost(equity, old_position, new_position)

    if old_position != new_position:
        n_trades += 1

    retorno_acumulado = (equity / ENV["initial_capital"] - 1.0) * 100.0

    # 7. Log
    new_row = {
        "timestamp":            pd.Timestamp.now(tz="UTC").floor("h"),
        "preco":                round(current_price, 2),
        "decisao":              decision_str,
        "posicao_anterior":     old_position,
        "posicao_atual":        new_position,
        "equity":               round(equity, 4),
        "retorno_candle_pct":   round(ret_candle, 4),
        "retorno_acumulado_pct":round(retorno_acumulado, 4),
        "n_trades":             n_trades,
    }
    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
    save_log(log)

    # 8. Saida no console (visivel nos logs do GitHub Actions)
    trade_flag = " *** TRADE ***" if old_position != new_position else ""
    print(f"\n  Preco atual:     ${current_price:,.2f}")
    print(f"  Posicao anterior:{POSITION_MAP[old_position]}")
    print(f"  Decisao agente:  {decision_str}{trade_flag}")
    print(f"  Equity simulada: ${equity:,.2f}")
    print(f"  Retorno acum.:   {retorno_acumulado:+.2f}%")
    print(f"  Trades totais:   {n_trades}")
    print(f"\n  Log salvo em:    {LOG_PATH.resolve()}")

    # Alerta de trade para facilitar leitura do log
    if old_position != new_position:
        print(f"\n  [!] MUDANCA DE POSICAO: {POSITION_MAP[old_position]} -> {POSITION_MAP[new_position]}")


if __name__ == "__main__":
    run()
