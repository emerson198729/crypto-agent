"""
Construção do vetor de observação para o agente RL.

Princípios aplicados:
- Todas as features são adimensionais (sem unidade de preço absoluta).
  Redes neurais não conseguem generalizar quando o preço do BTC vai de
  30k para 90k — mas retornos percentuais e osciladores são estáveis.
- Nenhuma feature usa dados futuros (sem look-ahead bias).
- NaN são preenchidos com 0.0 — o agente trata o início da série como
  mercado "neutro", o que é conservador e seguro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta


FEATURE_COLS: list[str] = [
    "ret_1",      # retorno 1 candle
    "ret_4",      # retorno 4 candles (4h em 1h tf)
    "ret_24",     # retorno 24 candles (1 dia)
    "ret_168",    # retorno 168 candles (7 dias) — regime de curto prazo
    "ret_720",    # retorno 720 candles (30 dias) — regime de medio prazo
    "rsi",        # RSI(14) normalizado para [-1, 1]
    "bb_pct",     # Bollinger %B: posicao do preco dentro das bandas
    "atr_pct",    # ATR relativo ao preco (volatilidade adimensional)
    "vol_ratio",  # volume atual / media 24 candles (acima/abaixo da media)
    "macd_hist",  # histograma MACD normalizado pelo preco
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe DataFrame OHLCV e retorna o mesmo DataFrame com colunas de features.

    Todas as colunas novas são adicionadas em uma cópia.
    Os NaN do início (warm-up dos indicadores) são zerados.
    """
    out = df.copy()
    close = out["close"]

    # Retornos em log: mais estáveis estatisticamente que retorno simples
    # e simétricos (subida de 10% + queda de 10% = 0, não -1%)
    log_ret = np.log(close / close.shift(1))
    out["ret_1"]   = log_ret
    out["ret_4"]   = np.log(close / close.shift(4))
    out["ret_24"]  = np.log(close / close.shift(24))
    out["ret_168"] = np.log(close / close.shift(168))  # 7 dias de regime
    out["ret_720"] = np.log(close / close.shift(720))  # 30 dias de regime

    # RSI normalizado: [0,100] → [-1,1]
    # Assim 50 (neutro) vira 0, sobrecomprado vira +1, sobrevendido vira -1
    rsi_raw = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    out["rsi"] = (rsi_raw - 50.0) / 50.0

    # Bollinger %B: onde o preço está dentro das bandas
    # 0 = na banda inferior, 1 = na banda superior, 0.5 = na média
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_lower = bb.bollinger_lband()
    bb_upper = bb.bollinger_hband()
    band_width = bb_upper - bb_lower
    out["bb_pct"] = np.where(band_width > 0, (close - bb_lower) / band_width, 0.5)

    # ATR relativo: volatilidade como fração do preço
    # Permite comparar períodos de preço muito diferente (BTC a 30k vs 90k)
    atr_raw = ta.volatility.AverageTrueRange(
        high=out["high"], low=out["low"], close=close, window=14
    ).average_true_range()
    out["atr_pct"] = atr_raw / close

    # Volume relativo: quanto acima/abaixo da média de 24 candles
    # Log para comprimir outliers (volume 10x a média vira 2.3, não 10)
    vol_mean = out["volume"].rolling(24, min_periods=1).mean()
    out["vol_ratio"] = np.log((out["volume"] / vol_mean).clip(lower=1e-6))

    # MACD histograma normalizado pelo preco
    # Captura momentum sem depender do nivel absoluto de preco
    macd = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    out["macd_hist"] = macd.macd_diff() / close

    # Preenche NaN do warm-up com 0 — tratamento conservador
    out[FEATURE_COLS] = out[FEATURE_COLS].fillna(0.0)

    return out


def get_observation_window(
    df: pd.DataFrame,
    current_idx: int,
    window_size: int,
) -> np.ndarray:
    """
    Extrai a janela de observação do agente no passo `current_idx`.

    Retorna array 1D de shape (window_size * n_features,) — formato
    esperado pela política MLP do stable-baselines3.

    O array é clipado em [-5, 5] para evitar que outliers extremos
    (ex: flash crash) destabilizem os gradientes durante o treino.
    """
    start = current_idx - window_size
    window = df[FEATURE_COLS].iloc[start:current_idx].values  # (window, n_features)
    flat = window.flatten().astype(np.float32)
    return np.clip(flat, -5.0, 5.0)
