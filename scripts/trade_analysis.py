"""
Analise detalhada por trade — V14 walk-forward.

Para cada janela, carrega o modelo salvo, roda o episodio deterministico
e extrai cada trade individualmente (entrada, saida, retorno, duracao).

Metricas reportadas:
  - Win rate por trade (% de trades lucrativos)
  - Profit factor (soma ganhos / soma perdas)
  - Retorno medio: trades ganhos vs perdas
  - Duracao media das posicoes
  - Distribuicao long vs short
  - Assertividade direcional (posicao correta vs movimento de mercado)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from config import DATA, ENV, WALK_FORWARD
from data.loader import fetch_ohlcv
from env.trading_env import TradingEnv
from features.builder import build_features


# ------------------------------------------------------------------
# Extrai trades individuais de um episodio
# ------------------------------------------------------------------

def extract_trades(result: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Percorre o log passo-a-passo e extrai cada trade completo.

    Um trade = abertura de posicao (entrada) ate fechamento (saida para flat
    ou inversao de posicao).

    Retorna DataFrame com uma linha por trade:
      side, entry_step, exit_step, duration,
      entry_price, exit_price, pct_return, won
    """
    trades = []
    in_trade   = False
    entry_step = None
    entry_price = None
    side       = None

    close_prices = df_test["close"].reset_index(drop=True)

    for i, row in result.iterrows():
        pos = row["position"]
        step = row["step"]

        # Preco do candle atual (alinhado ao step do env)
        price_idx = min(int(step), len(close_prices) - 1)
        price = float(close_prices.iloc[price_idx])

        if not in_trade:
            if pos != 0:
                # Abertura de posicao
                in_trade    = True
                entry_step  = step
                entry_price = price
                side        = pos  # +1 long, -1 short
        else:
            if pos == 0 or pos != side:
                # Fechamento: foi para flat ou inverteu
                exit_step  = step
                exit_price = price
                duration   = exit_step - entry_step

                if entry_price > 0:
                    raw_ret = (exit_price - entry_price) / entry_price
                    trade_ret = side * raw_ret  # long: normal, short: invertido
                else:
                    trade_ret = 0.0

                trades.append({
                    "side":         "LONG" if side == 1 else "SHORT",
                    "entry_step":   entry_step,
                    "exit_step":    exit_step,
                    "duration":     duration,
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "pct_return":   trade_ret * 100,
                    "won":          trade_ret > 0,
                })

                # Se inverteu (nao foi para flat), abre novo trade
                if pos != 0:
                    in_trade    = True
                    entry_step  = step
                    entry_price = price
                    side        = pos
                else:
                    in_trade = False

    # Fecha trade aberto no final do episodio
    if in_trade and entry_price > 0:
        last_step  = int(result["step"].iloc[-1])
        last_price = float(close_prices.iloc[min(last_step, len(close_prices)-1)])
        raw_ret    = (last_price - entry_price) / entry_price
        trade_ret  = side * raw_ret
        trades.append({
            "side":        "LONG" if side == 1 else "SHORT",
            "entry_step":  entry_step,
            "exit_step":   last_step,
            "duration":    last_step - entry_step,
            "entry_price": entry_price,
            "exit_price":  last_price,
            "pct_return":  trade_ret * 100,
            "won":         trade_ret > 0,
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ------------------------------------------------------------------
# Relatorio por janela
# ------------------------------------------------------------------

def report_window(w: int, trades_df: pd.DataFrame, result: pd.DataFrame,
                  bh_ret: float) -> dict:
    if trades_df.empty:
        print(f"  Janela {w}: nenhum trade registrado.")
        return {}

    n       = len(trades_df)
    winners = trades_df[trades_df["won"]]
    losers  = trades_df[~trades_df["won"]]

    win_rate     = len(winners) / n * 100
    avg_win      = winners["pct_return"].mean() if len(winners) > 0 else 0.0
    avg_loss     = losers["pct_return"].mean()  if len(losers)  > 0 else 0.0
    profit_factor = (winners["pct_return"].sum() /
                     abs(losers["pct_return"].sum())
                     if len(losers) > 0 and losers["pct_return"].sum() != 0
                     else float("inf"))

    avg_duration = trades_df["duration"].mean()
    n_long  = (trades_df["side"] == "LONG").sum()
    n_short = (trades_df["side"] == "SHORT").sum()

    # Assertividade direcional: trade na direcao correta do mercado?
    # Definimos "correto" como: LONG quando preco subiu no trade, SHORT quando caiu
    direcional = (trades_df["pct_return"] > 0).sum() / n * 100  # == win_rate gross

    # Retorno acumulado da equity no episodio
    equity = result["equity"]
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    print(f"\n  Janela {w}:")
    print(f"    Retorno agente:      {total_ret:+.1f}%  |  B&H: {bh_ret:+.1f}%")
    print(f"    Total de trades:     {n}")
    print(f"    Long / Short:        {n_long} / {n_short}")
    print(f"    ---------------------------------")
    print(f"    Win rate:            {win_rate:.1f}%  ({len(winners)}W / {len(losers)}L)")
    print(f"    Profit Factor:       {profit_factor:.2f}")
    print(f"    Retorno medio ganho: {avg_win:+.2f}%")
    print(f"    Retorno medio perda: {avg_loss:+.2f}%")
    print(f"    Duracao media:       {avg_duration:.0f} candles ({avg_duration/24:.1f}h)")
    print(f"    Assertiv. direcional:{direcional:.1f}%")

    return {
        "janela": w, "n_trades": n, "win_rate": win_rate,
        "profit_factor": profit_factor, "avg_win": avg_win,
        "avg_loss": avg_loss, "avg_duration": avg_duration,
        "direcional": direcional, "total_ret": total_ret,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  ANALISE DETALHADA POR TRADE — V14")
    print("=" * 60)

    raw = fetch_ohlcv(
        exchange_name=DATA["exchange"],
        symbol=DATA["symbol"],
        timeframe=DATA["timeframe"],
        days=DATA["days_train"] + DATA["days_test"],
    )
    df = build_features(raw).reset_index(drop=True)

    n_windows  = WALK_FORWARD["n_windows"]
    slice_size = len(df) // (n_windows + 1)
    slices = [df.iloc[i * slice_size:(i + 1) * slice_size].reset_index(drop=True)
              for i in range(n_windows + 1)]

    model_dir = Path(WALK_FORWARD["model_dir"])
    summaries = []

    # B&H por janela (calculado sobre o slice de teste)
    bh_rets = []
    for w in range(n_windows):
        df_test  = slices[w + 1]
        close    = df_test["close"].iloc[ENV["window_size"]:]
        bh_rets.append((close.iloc[-1] / close.iloc[0] - 1) * 100)

    for w in range(n_windows):
        model_path = model_dir / f"wf_window_{w + 1}.zip"
        if not model_path.exists():
            print(f"  [!] Modelo nao encontrado: {model_path}")
            continue

        model   = PPO.load(str(model_path))
        df_test = slices[w + 1]

        # Episodio deterministico
        env  = TradingEnv(df_test, ENV)
        obs, _ = env.reset()
        records = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            records.append({"step": info["step"], "equity": info["equity"],
                            "position": info["position"]})
            if terminated or truncated:
                break
        result = pd.DataFrame(records)

        trades_df = extract_trades(result, df_test)
        stats = report_window(w + 1, trades_df, result, bh_rets[w])
        if stats:
            summaries.append(stats)

    # ------------------------------------------------------------------
    # Resumo agregado
    # ------------------------------------------------------------------
    if summaries:
        df_sum = pd.DataFrame(summaries)
        print("\n" + "=" * 60)
        print("  RESUMO AGREGADO — TODAS AS JANELAS")
        print("=" * 60)
        print(f"  Trades totais:          {df_sum['n_trades'].sum()}")
        print(f"  Win rate medio:         {df_sum['win_rate'].mean():.1f}%")
        print(f"  Profit Factor medio:    {df_sum['profit_factor'].mean():.2f}")
        print(f"  Retorno medio (ganhos): {df_sum['avg_win'].mean():+.2f}%")
        print(f"  Retorno medio (perdas): {df_sum['avg_loss'].mean():+.2f}%")
        print(f"  Duracao media:          {df_sum['avg_duration'].mean():.0f} candles")
        print(f"  Assertiv. direcional:   {df_sum['direcional'].mean():.1f}%")
        print("=" * 60)

        # Grafico win/loss por janela
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(df_sum["janela"], df_sum["win_rate"], color=["green" if r > 50 else "red"
                    for r in df_sum["win_rate"]])
        axes[0].axhline(50, color="gray", ls="--", lw=1)
        axes[0].set_title("Win Rate por Janela (%)")
        axes[0].set_xlabel("Janela")
        axes[0].set_ylabel("%")
        axes[0].set_ylim(0, 100)

        axes[1].bar(df_sum["janela"], df_sum["profit_factor"],
                    color=["green" if pf > 1 else "red" for pf in df_sum["profit_factor"]])
        axes[1].axhline(1.0, color="gray", ls="--", lw=1)
        axes[1].set_title("Profit Factor por Janela")
        axes[1].set_xlabel("Janela")
        axes[1].set_ylabel("PF")

        plt.tight_layout()
        out = "results/trade_analysis.png"
        Path("results").mkdir(exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Grafico salvo em {out}")


if __name__ == "__main__":
    main()
