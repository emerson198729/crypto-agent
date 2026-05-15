"""
Validação do ambiente antes de iniciar o treino.

Roda três verificações:

  1. gymnasium.utils.env_checker  — garante que o ambiente segue o
     contrato da API (observation/action spaces, reset, step).

  2. Episódio com ações aleatórias  — smoke test: garante que o loop
     roda sem exceções do início ao fim.

  3. Inspeção da observação  — imprime shape, min, max e NaN count
     para detectar features mal normalizadas cedo.

Rode ANTES de python -m agent.train para economizar horas de debug.

Como rodar (da pasta crypto-agent/):
  python scripts/sanity_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from gymnasium.utils.env_checker import check_env

from config import DATA, ENV
from data.loader import fetch_ohlcv, train_test_split
from env.trading_env import TradingEnv
from features.builder import build_features


def main() -> None:
    print("=" * 55)
    print("  SANITY CHECK — TradingEnv")
    print("=" * 55)

    # --- Dados ---
    print("\n[1/4] Baixando / carregando dados...")
    raw = fetch_ohlcv(
        exchange_name=DATA["exchange"],
        symbol=DATA["symbol"],
        timeframe=DATA["timeframe"],
        days=60,  # 60 dias suficiente para o check
    )

    print("[2/4] Construindo features...")
    df = build_features(raw)
    df_train, _ = train_test_split(df, test_days=14)

    env = TradingEnv(df_train, ENV)

    # --- Verificação da API Gymnasium ---
    print("\n[3/4] Verificando contrato da API Gymnasium...")
    try:
        check_env(env, warn=True, skip_render_check=True)
        print("      OK — ambiente compatível com Gymnasium")
    except Exception as exc:
        print(f"      ERRO: {exc}")
        sys.exit(1)

    # --- Episódio aleatório completo ---
    print("\n[4/4] Rodando episódio com ações aleatórias...")
    obs, _ = env.reset()
    n_steps = 0
    total_reward = 0.0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1
        if terminated or truncated:
            break

    print(f"      Passos executados:  {n_steps}")
    print(f"      Recompensa total:   {total_reward:.4f}")
    print(f"      Equity final:       ${info['equity']:.2f}")
    print(f"      Drawdown final:     {info['drawdown'] * 100:.2f}%")

    # --- Inspeção da observação ---
    obs_arr = np.array(obs)
    nan_count = int(np.isnan(obs_arr).sum())
    print(f"\n      Shape observação:   {obs_arr.shape}")
    print(f"      Min / Max:          {obs_arr.min():.4f} / {obs_arr.max():.4f}")
    print(f"      NaN count:          {nan_count}")

    if nan_count > 0:
        print("\n  AVISO: observação contém NaN — revisar features/builder.py")
        sys.exit(1)

    if abs(obs_arr.max()) > 10 or abs(obs_arr.min()) > 10:
        print("\n  AVISO: valores fora de [-10, 10] — pode desestabilizar o treino")

    print("\n" + "=" * 55)
    print("  Tudo ok — pode rodar: python -m agent.train")
    print("=" * 55)


if __name__ == "__main__":
    main()
