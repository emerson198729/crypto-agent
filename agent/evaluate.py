"""
Avaliação do agente treinado no conjunto de teste out-of-sample.

O que este script responde:
  - O agente gerou retorno positivo no período que nunca viu?
  - Bateu o Buy & Hold?
  - Qual o Sharpe, drawdown e win rate?

Só confie nos números daqui — nunca nos números do treino,
que são sempre otimistas (o modelo viu esses dados).

Como rodar (da pasta crypto-agent/):
  python -m agent.evaluate --model models/best/best_model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from config import DATA, ENV
from data.loader import fetch_ohlcv, train_test_split
from env.trading_env import TradingEnv
from features.builder import build_features


# ------------------------------------------------------------------
# Métricas
# ------------------------------------------------------------------

def _sharpe(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def _profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _win_rate(returns: pd.Series) -> float:
    operated = returns[returns != 0]
    if len(operated) == 0:
        return 0.0
    return float((operated > 0).mean())


# ------------------------------------------------------------------
# Execução determinística do agente
# ------------------------------------------------------------------

def run_episode(model: PPO, env: TradingEnv) -> pd.DataFrame:
    """
    Roda um episódio completo com política determinística.

    Determinístico = o agente escolhe sempre a ação com maior probabilidade,
    sem exploração aleatória. É assim que ele deve operar em produção.
    """
    obs, _ = env.reset()
    records = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        records.append({
            "step":     info["step"],
            "equity":   info["equity"],
            "position": info["position"],
            "action":   int(action),
            "reward":   reward,
        })

        if terminated or truncated:
            break

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# Relatório
# ------------------------------------------------------------------

def print_report(result: pd.DataFrame, df_test: pd.DataFrame) -> None:
    equity = pd.Series(result["equity"].values)
    # Retornos passo a passo (log-return)
    log_ret = np.log(equity / equity.shift(1)).fillna(0.0)

    # Trades = mudanças de posição
    n_trades = int(result["position"].diff().abs().sum())

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    bh_ret = (df_test["close"].iloc[-1] / df_test["close"].iloc[0] - 1) * 100

    print("\n" + "=" * 50)
    print("  AVALIAÇÃO OUT-OF-SAMPLE")
    print("=" * 50)
    print(f"  Retorno agente:    {total_ret:+.2f}%")
    print(f"  Retorno Buy&Hold:  {bh_ret:+.2f}%")
    print(f"  Alpha:             {total_ret - bh_ret:+.2f}pp")
    print(f"  Sharpe (anual):    {_sharpe(log_ret):.2f}")
    print(f"  Max Drawdown:      {_max_drawdown(equity) * 100:.2f}%")
    print(f"  Profit Factor:     {_profit_factor(log_ret):.2f}")
    print(f"  Win Rate:          {_win_rate(log_ret) * 100:.2f}%")
    print(f"  Total de trades:   {n_trades}")
    print(f"  Equity final:      ${equity.iloc[-1]:,.2f}")
    print("=" * 50)


def plot_results(result: pd.DataFrame, df_test: pd.DataFrame, save_path: str = "evaluation.png") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    steps = result["step"].values
    prices = df_test["close"].iloc[: len(steps)].values

    # Painel 1: preço + regiões compradas
    axes[0].plot(steps, prices, color="gray", alpha=0.7, linewidth=1, label="BTC/USDT")
    axes[0].fill_between(
        steps, prices.min(), prices.max(),
        where=(result["position"].values == 1),
        alpha=0.15, color="green", label="Comprado",
    )
    axes[0].set_title("Preço e períodos comprados")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # Painel 2: equity curve vs Buy & Hold
    bh = (prices / prices[0]) * ENV["initial_capital"]
    axes[1].plot(steps, result["equity"].values, color="royalblue", linewidth=1.5, label="Agente PPO")
    axes[1].plot(steps, bh, color="orange", linestyle="--", linewidth=1.5, label="Buy & Hold")
    axes[1].set_title("Equity Curve — out-of-sample")
    axes[1].set_ylabel("USDT")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    # Painel 3: drawdown
    eq = pd.Series(result["equity"].values)
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    axes[2].fill_between(steps, dd.values, 0, color="red", alpha=0.4)
    axes[2].plot(steps, dd.values, color="red", linewidth=0.8)
    axes[2].set_title("Drawdown (%)")
    axes[2].set_ylabel("%")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[grafico] Salvo em {save_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def evaluate(model_path: str) -> None:
    print("[dados] Carregando...")
    raw = fetch_ohlcv(
        exchange_name=DATA["exchange"],
        symbol=DATA["symbol"],
        timeframe=DATA["timeframe"],
        days=DATA["days_train"] + DATA["days_test"],
    )
    df = build_features(raw)
    _, df_test = train_test_split(df, test_days=DATA["days_test"])

    print(f"[modelo] Carregando {model_path}...")
    model = PPO.load(model_path)

    env = TradingEnv(df_test, ENV)

    print("[avaliacao] Rodando episodio deterministico...")
    result = run_episode(model, env)

    print_report(result, df_test)
    plot_results(result, df_test, save_path="evaluation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/best/best_model",
        help="Caminho para o modelo salvo (sem .zip)",
    )
    args = parser.parse_args()
    evaluate(args.model)
