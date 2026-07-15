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
import ta
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
# Overlay de risco (camada B do plano) — independente do modelo
# ------------------------------------------------------------------
# Stop-loss por posicao: se o prejuizo nao-realizado da posicao aberta
# passar deste limite, o runner FORCA saida para FLAT, ignorando o modelo.
# Ataca diretamente o vies de "segurar LONG durante a queda".
STOP_LOSS_PCT  = 0.04    # 4% de prejuizo na posicao -> corta

# Apos um stop, fica FLAT por COOLDOWN_BARS candles antes de reentrar.
# Evita reentrar na "faca caindo" logo apos ser stopado.
COOLDOWN_BARS  = 3

# Filtro de chop (mercado sem tendencia definida): quando ADX(14) < threshold,
# bloqueia a ABERTURA de nova posicao direcional — mercado lateral gera
# whipsaw caro para um agente trend-follower. Validado via shadow backtest
# (scripts/shadow_backtest.py) contra dados reais da Kraken (mesma fonte
# usada aqui): reduziu perda de -13.9% para -9.2% e MDD de -17.7% para
# -11.1% no periodo de chop de jun-jul/2026. Fechar para FLAT continua
# sempre permitido — o filtro so impede ENTRAR em LONG/SHORT sem tendencia.
CHOP_ADX_THRESHOLD = 20

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
# Reconstrucao de estado a partir do log
# ------------------------------------------------------------------

def reconstruct_entry_price(log: pd.DataFrame, position: int) -> float:
    """Preco em que a posicao atualmente aberta foi iniciada."""
    if log.empty or position == 0:
        return 0.0
    # Procura de tras pra frente a linha onde a posicao mudou para o valor atual
    for i in range(len(log) - 1, -1, -1):
        row = log.iloc[i]
        if int(row["posicao_atual"]) == position and int(row["posicao_anterior"]) != position:
            return float(row["preco"])
    # Nao achou transicao explicita — usa o preco mais antigo com essa posicao
    same = log[log["posicao_atual"] == position]
    return float(same.iloc[0]["preco"]) if not same.empty else 0.0


def steps_since_entry(log: pd.DataFrame, position: int) -> int:
    """Quantos candles a posicao atual ja durou (para time_in_position)."""
    if log.empty or position == 0:
        return 0
    count = 0
    for i in range(len(log) - 1, -1, -1):
        if int(log.iloc[i]["posicao_atual"]) == position:
            count += 1
            if int(log.iloc[i]["posicao_anterior"]) != position:
                break
        else:
            break
    return count


# ------------------------------------------------------------------
# Runner principal — processa TODOS os candles perdidos (catch-up)
# ------------------------------------------------------------------

def run() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*55}")
    print(f"  PAPER TRADING — {now}")
    print(f"{'='*55}")

    from features.builder import FEATURE_COLS
    window       = ENV["window_size"]
    initial_cap  = float(ENV["initial_capital"])
    current_hour = pd.Timestamp.now(tz="UTC").floor("h")

    # 1. Modelo
    if not MODEL_PATH.exists():
        print(f"[ERRO] Modelo nao encontrado: {MODEL_PATH}")
        sys.exit(1)
    model = PPO.load(str(MODEL_PATH))
    print(f"[modelo] carregado de {MODEL_PATH}")

    # 2. Dados ao vivo (mantem timestamp no indice)
    df_raw = fetch_live_candles()
    df     = build_features(df_raw)              # indice = timestamp
    feats  = df[FEATURE_COLS].values.astype("float32")
    closes = df["close"].values.astype(float)
    times  = df.index

    # ADX(14) para o filtro de chop — calculado fora do vetor de features do
    # modelo (nao altera a observacao, e um overlay externo de decisao)
    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx().fillna(0.0).values

    # 3. Estado inicial a partir do log
    log = load_log()
    old_position, equity, n_trades = get_current_state(log)
    entry_price = reconstruct_entry_price(log, old_position)
    steps_in_pos = steps_since_entry(log, old_position)
    peak_equity  = float(log["equity"].max()) if not log.empty else equity
    last_ts      = pd.to_datetime(log["timestamp"].iloc[-1], utc=True) if not log.empty else None

    # Cooldown: ultimas decisoes registradas (para detectar STOP recente)
    recent_decisions = (
        list(log["decisao"].tail(COOLDOWN_BARS)) if not log.empty else []
    )

    # 4. Seleciona candles a processar: novos (apos last_ts) e ja FECHADOS (< hora atual)
    new_rows = []
    for i in range(max(window, 1), len(closes)):
        ts = times[i]
        if ts >= current_hour:
            continue                       # candle ainda em formacao — ignora
        if last_ts is not None and ts <= last_ts:
            continue                       # ja registrado
        if last_ts is None and i < len(closes) - 1:
            continue                       # primeiro run: processa so o ultimo fechado
        new_rows.append(i)

    if not new_rows:
        print("[runner] Nenhum candle novo fechado para processar. Saindo.")
        return

    print(f"[runner] Processando {len(new_rows)} candle(s) novo(s)...")

    # 5. Loop de catch-up — um passo por candle perdido
    for i in new_rows:
        price = closes[i]
        prev  = closes[i - 1]
        ret   = (price - prev) / prev if prev > 0 else 0.0

        # 5a. Realiza o retorno do candle que fechou, sob a posicao que vinha aberta
        equity *= (1.0 + old_position * ret)

        # 5b. Observacao: janela terminando neste candle + estado de conta
        obs_window = np.clip(feats[i - window + 1:i + 1].flatten(), -5.0, 5.0)
        unrealized = (
            old_position * (price - entry_price) / entry_price
            if entry_price > 0 and old_position != 0 else 0.0
        )
        drawdown  = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        time_norm = min(steps_in_pos / window, 1.0)
        obs = np.concatenate([
            obs_window,
            np.array([float(old_position), unrealized, drawdown, time_norm], dtype="float32"),
        ])

        # 5c. Decisao do modelo
        action, _    = model.predict(obs, deterministic=True)
        model_pos    = ACTION_TO_POS[int(action)]
        new_position = model_pos
        decision_str = ACTION_MAP[int(action)]

        # 5d. OVERLAY DE RISCO (independente do modelo)
        # Cooldown: se houve STOP nos ultimos COOLDOWN_BARS candles, fica FLAT
        in_cooldown = "STOP" in recent_decisions[-COOLDOWN_BARS:]
        if in_cooldown:
            new_position = 0
            decision_str = "FLAT"

        # Stop-loss: posicao aberta com prejuizo acima do limite -> corta
        elif old_position != 0 and unrealized < -STOP_LOSS_PCT:
            new_position = 0
            decision_str = "STOP"

        # Chop filter: bloqueia ABERTURA de posicao direcional sem tendencia.
        # Fechar para FLAT continua sempre permitido. Validado em backtest
        # contra dados reais da Kraken (ver CHOP_ADX_THRESHOLD acima).
        elif (new_position != old_position and new_position != 0
              and adx[i] < CHOP_ADX_THRESHOLD):
            new_position = old_position if old_position != 0 else 0
            if new_position == 0:
                decision_str = "CHOP_SKIP"

        # 5e. Custo de transacao se mudou de posicao
        if new_position != old_position:
            n_trans = 1 if (old_position == 0 or new_position == 0) else 2
            equity *= (1.0 - n_trans * (ENV["fee_rate"] + ENV["slippage_rate"]))
            n_trades += 1
            entry_price  = price
            steps_in_pos = 0
        else:
            steps_in_pos += 1

        peak_equity = max(peak_equity, equity)
        retorno_acum = (equity / initial_cap - 1.0) * 100.0

        row_data = {
            "timestamp":             ts_floor(times[i]),
            "preco":                 round(price, 2),
            "decisao":               decision_str,
            "posicao_anterior":      old_position,
            "posicao_atual":         new_position,
            "equity":                round(equity, 4),
            "retorno_candle_pct":    round(ret * 100.0, 4),
            "retorno_acumulado_pct": round(retorno_acum, 4),
            "n_trades":              n_trades,
        }
        log = pd.concat([log, pd.DataFrame([row_data])], ignore_index=True)

        recent_decisions.append(decision_str)
        old_position = new_position

    save_log(log)

    # 6. Resumo da ultima decisao
    last = log.iloc[-1]
    print(f"\n  Candles processados: {len(new_rows)}")
    print(f"  Preco atual:     ${float(last['preco']):,.2f}")
    print(f"  Decisao final:   {last['decisao']}")
    print(f"  Posicao atual:   {POSITION_MAP[int(last['posicao_atual'])]}")
    print(f"  Equity simulada: ${float(last['equity']):,.2f}")
    print(f"  Retorno acum.:   {float(last['retorno_acumulado_pct']):+.2f}%")
    print(f"  Trades totais:   {int(last['n_trades'])}")


def ts_floor(ts) -> pd.Timestamp:
    """Normaliza o timestamp do candle para a hora cheia em UTC."""
    return pd.Timestamp(ts).tz_convert("UTC").floor("h")


if __name__ == "__main__":
    run()
