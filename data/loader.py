"""
Download e cache de dados OHLCV via ccxt.

Cache em parquet evita re-download a cada execução.
O split treino/teste é feito por data, nunca aleatório —
embaralhar séries temporais é look-ahead bias disfarçado.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(
    exchange_name: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 365,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Baixa OHLCV e retorna DataFrame indexado por timestamp UTC.

    Na primeira chamada faz o download e cacheia em parquet.
    Nas seguintes, lê do cache — muito mais rápido.
    """
    cache_name = f"{exchange_name}_{symbol.replace('/', '_')}_{timeframe}_{days}d.parquet"
    cache_file = DATA_DIR / cache_name

    if use_cache and cache_file.exists():
        print(f"[cache] {cache_file.name}")
        return pd.read_parquet(cache_file)

    if not hasattr(ccxt, exchange_name):
        raise ValueError(f"Exchange '{exchange_name}' não suportada pelo ccxt.")

    exchange: ccxt.Exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})

    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_candles: list[list] = []
    limit = 1000

    print(f"[download] {exchange_name} {symbol} {timeframe} — últimos {days} dias")

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        raise RuntimeError("Nenhum dado retornado pela exchange.")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df.to_parquet(cache_file)
    print(f"[ok] {len(df)} candles salvos em {cache_file.name}")

    return df


def train_test_split(df: pd.DataFrame, test_days: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o DataFrame em treino e teste por data.

    O teste sempre é o período mais recente — simula deploy real.
    NUNCA embaralhar séries temporais: vazar dados futuros invalida
    qualquer métrica de backtest.
    """
    cutoff = df.index[-1] - pd.Timedelta(days=test_days)
    train = df[df.index <= cutoff].copy()
    test = df[df.index > cutoff].copy()

    print(f"[split] treino: {len(train)} candles ({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"[split] teste:  {len(test)} candles ({test.index[0].date()} -> {test.index[-1].date()})")

    return train, test
