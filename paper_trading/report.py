"""
Relatorio semanal automatico do paper trading.

Executado toda segunda-feira pelo GitHub Actions.
Imprime um resumo da semana anterior no log do workflow.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

LOG_PATH = Path(__file__).parent / "log.csv"


def compute_bh(log: pd.DataFrame) -> float:
    """Buy & Hold retorno no mesmo periodo."""
    if len(log) < 2:
        return 0.0
    return (log["preco"].iloc[-1] / log["preco"].iloc[0] - 1.0) * 100.0


def profit_factor(log: pd.DataFrame) -> float:
    gains  = log[log["retorno_candle_pct"] > 0]["retorno_candle_pct"].sum()
    losses = log[log["retorno_candle_pct"] < 0]["retorno_candle_pct"].abs().sum()
    return round(gains / losses, 2) if losses > 0 else float("inf")


def win_rate(log: pd.DataFrame) -> float:
    active = log[log["posicao_atual"] != 0]
    if active.empty:
        return 0.0
    return round((active["retorno_candle_pct"] > 0).mean() * 100, 1)


def print_report(log: pd.DataFrame, periodo: str) -> None:
    if log.empty:
        print("Sem dados suficientes para relatorio.")
        return

    retorno_agente = float(log["retorno_acumulado_pct"].iloc[-1])
    retorno_bh     = compute_bh(log)
    alpha          = retorno_agente - retorno_bh
    n_trades       = int(log["n_trades"].iloc[-1] - log["n_trades"].iloc[0])
    pf             = profit_factor(log)
    wr             = win_rate(log)
    equity_final   = float(log["equity"].iloc[-1])
    equity_inicial = float(log["equity"].iloc[0])
    mdd            = float(
        ((log["equity"] - log["equity"].cummax()) / log["equity"].cummax()).min() * 100
    )

    pos_long  = (log["posicao_atual"] == 1).mean() * 100
    pos_short = (log["posicao_atual"] == -1).mean() * 100
    pos_flat  = (log["posicao_atual"] == 0).mean() * 100

    print("=" * 55)
    print(f"  RELATORIO PAPER TRADING — {periodo}")
    print("=" * 55)
    print(f"  Periodo:          {log['timestamp'].iloc[0]} -> {log['timestamp'].iloc[-1]}")
    print(f"  Candles:          {len(log)}")
    print()
    print(f"  Retorno agente:   {retorno_agente:+.2f}%")
    print(f"  Buy & Hold:       {retorno_bh:+.2f}%")
    print(f"  Alpha:            {alpha:+.2f}pp")
    print()
    print(f"  Equity inicial:   ${equity_inicial:,.2f}")
    print(f"  Equity final:     ${equity_final:,.2f}")
    print(f"  Max Drawdown:     {mdd:.2f}%")
    print()
    print(f"  Trades no periodo:{n_trades}")
    print(f"  Win Rate:         {wr}%")
    print(f"  Profit Factor:    {pf}")
    print()
    print(f"  Long:  {pos_long:.1f}%  |  Short: {pos_short:.1f}%  |  Flat: {pos_flat:.1f}%")
    print("=" * 55)


def main() -> None:
    if not LOG_PATH.exists():
        print("[report] Log ainda nao existe. Aguardando primeiros dados.")
        return

    log_full = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])

    if log_full.empty:
        print("[report] Log vazio.")
        return

    # Relatorio semanal (ultimos 7 dias)
    cutoff_7d = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    log_7d    = log_full[log_full["timestamp"] >= cutoff_7d]
    print_report(log_7d, "ULTIMA SEMANA")

    print()

    # Relatorio total (desde o inicio)
    print_report(log_full, "TOTAL ACUMULADO")


if __name__ == "__main__":
    main()
