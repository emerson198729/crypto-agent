"""
Walk-forward training e avaliação.

Por que walk-forward?
  O problema do run anterior: treinamos em nov/23→nov/25 (bull market)
  e testamos em nov/25→mai/26 (bear market). O agente nunca viu queda
  durante o treino, então aprendeu "sempre compra" — inútil em baixa.

  Walk-forward resolve isso dividindo o histórico em janelas sequenciais.
  Para cada janela, o agente treina no passado e é testado no futuro
  imediato — exatamente como operaria na vida real.

Estrutura com 4 janelas (910 dias ÷ 5 fatias ≈ 182 dias cada):

  Fatia:  [0]     [1]     [2]     [3]     [4]
  Jan 1:  treino  teste
  Jan 2:  treino  treino  teste
  Jan 3:  treino  treino  treino  teste
  Jan 4:  treino  treino  treino  treino  teste

  "Expanding window": o conjunto de treino cresce a cada janela.
  Isso simula um trader que reutiliza todo o histórico disponível.

Como rodar (da pasta raiz):
  python crypto-agent/agent/walk_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # renderiza sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from config import DATA, ENV, WALK_FORWARD
from data.loader import fetch_ohlcv
from env.trading_env import TradingEnv
from features.builder import build_features


# ------------------------------------------------------------------
# Métricas
# ------------------------------------------------------------------

def _metrics(equity: pd.Series, periods_per_year: int = 365 * 24) -> dict:
    ret = equity.pct_change().fillna(0.0)
    total  = float(equity.iloc[-1] / equity.iloc[0] - 1) * 100
    std    = ret.std()
    sharpe = float(np.sqrt(periods_per_year) * ret.mean() / std) if std > 1e-8 else 0.0
    peak   = equity.cummax()
    mdd    = float(((equity - peak) / peak).min()) * 100
    gains  = ret[ret > 0].sum()
    losses = -ret[ret < 0].sum()
    pf     = float(gains / losses) if losses > 0 else float("inf")
    return {"total_pct": total, "sharpe": sharpe, "mdd_pct": mdd, "profit_factor": pf}


# ------------------------------------------------------------------
# Um episódio determinístico
# ------------------------------------------------------------------

def _run_episode(model: PPO, df: pd.DataFrame) -> pd.DataFrame:
    env = TradingEnv(df, ENV)
    obs, _ = env.reset()
    records = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        records.append({
            "step":     info["step"],
            "equity":   info["equity"],
            "position": info["position"],
            "action":   int(action),
        })
        if terminated or truncated:
            break
    return pd.DataFrame(records)


# ------------------------------------------------------------------
# Walk-forward
# ------------------------------------------------------------------

def walk_forward() -> None:
    print("=" * 60)
    print("  WALK-FORWARD TRAINING + AVALIAÇÃO")
    print("=" * 60)

    # Dados completos
    raw = fetch_ohlcv(
        exchange_name=DATA["exchange"],
        symbol=DATA["symbol"],
        timeframe=DATA["timeframe"],
        days=DATA["days_train"] + DATA["days_test"],
    )
    df = build_features(raw)
    df = df.reset_index(drop=True)

    n_windows = WALK_FORWARD["n_windows"]
    timesteps = WALK_FORWARD["timesteps_per_window"]

    # Divide em n_windows+1 fatias iguais
    slice_size = len(df) // (n_windows + 1)
    slices = [df.iloc[i * slice_size:(i + 1) * slice_size].reset_index(drop=True)
              for i in range(n_windows + 1)]

    model_dir = Path(WALK_FORWARD["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(WALK_FORWARD["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    window_results = []

    for w in range(n_windows):
        print(f"\n{'-'*60}")
        print(f"  Janela {w+1}/{n_windows}")

        # Treino: fatias 0..w concatenadas (expanding window)
        df_train = pd.concat(slices[:w + 1], ignore_index=True)
        # Teste: fatia imediatamente seguinte
        df_test  = slices[w + 1]

        date_train_start = raw.index[0 * slice_size].date() if hasattr(raw.index[0], 'date') else "?"
        print(f"  Treino: {len(df_train)} candles")
        print(f"  Teste:  {len(df_test)} candles")

        # Treino PPO
        from config import TRAIN
        vec_env = make_vec_env(
            lambda: Monitor(TradingEnv(df_train, ENV)),
            n_envs=TRAIN["n_envs"],
            seed=TRAIN["seed"] + w,
        )
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=TRAIN["learning_rate"],
            n_steps=TRAIN["n_steps"],
            batch_size=TRAIN["batch_size"],
            policy_kwargs=TRAIN["policy_kwargs"],
            ent_coef=TRAIN["ent_coef"],
            verbose=0,           # silencioso — só mostra progresso da janela
            seed=TRAIN["seed"] + w,
        )

        print(f"  Treinando {timesteps:,} timesteps...", end=" ", flush=True)
        model.learn(total_timesteps=timesteps, progress_bar=False)
        print("ok")

        model.save(str(model_dir / f"wf_window_{w + 1}"))

        # Avaliação determinística
        result = _run_episode(model, df_test)
        equity = pd.Series(result["equity"].values)

        # Buy & Hold no mesmo período
        close_test = df_test["close"].iloc[ENV["window_size"]:]
        bh_equity  = (close_test / close_test.iloc[0]) * ENV["initial_capital"]
        bh_equity  = bh_equity.reset_index(drop=True)

        m = _metrics(equity)
        m_bh = _metrics(bh_equity.reindex(equity.index, fill_value=bh_equity.iloc[-1]))

        n_trades = int(result["position"].diff().abs().sum())
        pos_long  = (result["position"] == 1).mean() * 100
        pos_short = (result["position"] == -1).mean() * 100
        pos_flat  = (result["position"] == 0).mean() * 100

        window_results.append({
            "janela": w + 1,
            "retorno_agente":  m["total_pct"],
            "retorno_bh":      m_bh["total_pct"],
            "alpha":           m["total_pct"] - m_bh["total_pct"],
            "sharpe":          m["sharpe"],
            "mdd":             m["mdd_pct"],
            "profit_factor":   m["profit_factor"],
            "n_trades":        n_trades,
            "pct_long":        pos_long,
            "pct_short":       pos_short,
            "pct_flat":        pos_flat,
            "equity":          equity,
            "bh_equity":       bh_equity,
        })

        print(f"  Retorno agente: {m['total_pct']:+.1f}%  |  "
              f"Buy&Hold: {m_bh['total_pct']:+.1f}%  |  "
              f"Alpha: {m['total_pct'] - m_bh['total_pct']:+.1f}pp  |  "
              f"Trades: {n_trades}  |  "
              f"L:{pos_long:.0f}% S:{pos_short:.0f}% F:{pos_flat:.0f}%")

    # ------------------------------------------------------------------
    # Relatório agregado
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  RESUMO WALK-FORWARD")
    print(f"{'='*60}")

    alphas   = [r["alpha"] for r in window_results]
    sharpes  = [r["sharpe"] for r in window_results]
    trades   = [r["n_trades"] for r in window_results]

    print(f"  Alpha médio:       {np.mean(alphas):+.2f}pp")
    print(f"  Alpha positivo:    {sum(a > 0 for a in alphas)}/{len(alphas)} janelas")
    print(f"  Sharpe médio:      {np.mean(sharpes):.2f}")
    print(f"  Trades por janela: {np.mean(trades):.0f}")
    print(f"{'='*60}")

    # Tabela por janela
    print(f"\n  {'Jan':>4} {'Agente':>8} {'B&H':>8} {'Alpha':>8} {'Sharpe':>7} {'MDD':>7} {'Trades':>7}")
    print("  " + "-" * 55)
    for r in window_results:
        print(f"  {r['janela']:>4} "
              f"{r['retorno_agente']:>+7.1f}% "
              f"{r['retorno_bh']:>+7.1f}% "
              f"{r['alpha']:>+7.1f}pp "
              f"{r['sharpe']:>7.2f} "
              f"{r['mdd']:>6.1f}% "
              f"{r['n_trades']:>7}")

    # ------------------------------------------------------------------
    # Gráfico consolidado
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(n_windows, 1, figsize=(14, 4 * n_windows), squeeze=False)

    for i, r in enumerate(window_results):
        ax = axes[i][0]
        steps = np.arange(len(r["equity"]))
        bh    = r["bh_equity"].reindex(range(len(r["equity"])),
                                        fill_value=r["bh_equity"].iloc[-1])

        ax.plot(steps, r["equity"].values,  color="royalblue", lw=1.5, label="Agente PPO")
        ax.plot(steps, bh.values,           color="orange", ls="--", lw=1.5, label="Buy & Hold")
        ax.fill_between(steps, ENV["initial_capital"], r["equity"].values,
                         where=r["equity"].values >= ENV["initial_capital"],
                         alpha=0.1, color="green")
        ax.fill_between(steps, ENV["initial_capital"], r["equity"].values,
                         where=r["equity"].values < ENV["initial_capital"],
                         alpha=0.1, color="red")
        ax.axhline(ENV["initial_capital"], color="gray", ls=":", lw=0.8)
        ax.set_title(
            f"Janela {r['janela']} — "
            f"Agente: {r['retorno_agente']:+.1f}%  |  "
            f"B&H: {r['retorno_bh']:+.1f}%  |  "
            f"Alpha: {r['alpha']:+.1f}pp  |  "
            f"Trades: {r['n_trades']}"
        )
        ax.set_ylabel("USDT")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = str(results_dir / "walk_forward.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafico salvo em {chart_path}")


if __name__ == "__main__":
    walk_forward()
