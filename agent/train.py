"""
Treino do agente PPO no ambiente de trading.

Fluxo completo:
  1. Baixa / carrega dados históricos em cache
  2. Constrói features normalizadas
  3. Divide treino/teste por data (nunca aleatório)
  4. Treina PPO com ambientes paralelos
  5. Avalia periodicamente no conjunto de validação
  6. Salva checkpoints e o melhor modelo

Como rodar (da pasta crypto-agent/):
  python -m agent.train
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garante que imports absolutos funcionam ao rodar como script
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from config import DATA, ENV, TRAIN
from data.loader import fetch_ohlcv, train_test_split
from env.trading_env import TradingEnv
from features.builder import build_features


def _make_env(df, config):
    """Factory que retorna uma função criadora de ambiente — padrão make_vec_env."""
    def _init():
        return Monitor(TradingEnv(df, config))
    return _init


def train() -> None:
    # ------------------------------------------------------------------
    # 1. Dados
    # ------------------------------------------------------------------
    print("=" * 60)
    print("CRYPTO AGENT — TREINO PPO")
    print("=" * 60)

    raw = fetch_ohlcv(
        exchange_name=DATA["exchange"],
        symbol=DATA["symbol"],
        timeframe=DATA["timeframe"],
        days=DATA["days_train"] + DATA["days_test"],
    )

    # ------------------------------------------------------------------
    # 2. Features
    # ------------------------------------------------------------------
    print("\n[features] Calculando indicadores...")
    df = build_features(raw)

    # ------------------------------------------------------------------
    # 3. Split treino / validação
    # ------------------------------------------------------------------
    # Separamos os últimos days_test como validação out-of-sample.
    # O modelo nunca vê esses dados durante o treino.
    df_train, df_val = train_test_split(df, test_days=DATA["days_test"])

    # ------------------------------------------------------------------
    # 4. Ambientes vetorizados
    # ------------------------------------------------------------------
    # n_envs ambientes paralelos aceleram coleta de experiência do PPO.
    # Cada um roda o mesmo dataset de treino de forma independente.
    print(f"\n[env] Criando {TRAIN['n_envs']} ambientes de treino paralelos...")
    vec_train = make_vec_env(
        _make_env(df_train, ENV),
        n_envs=TRAIN["n_envs"],
        seed=TRAIN["seed"],
    )

    # Ambiente de validação: único, determinístico, sem paralelismo
    env_val = Monitor(TradingEnv(df_val, ENV))

    # ------------------------------------------------------------------
    # 5. Modelo PPO
    # ------------------------------------------------------------------
    # MlpPolicy: rede MLP padrão — adequada para observações 1D (vetor flat)
    # A política aprende uma distribuição sobre ações dado o estado atual
    model = PPO(
        policy="MlpPolicy",
        env=vec_train,
        learning_rate=TRAIN["learning_rate"],
        n_steps=TRAIN["n_steps"],
        batch_size=TRAIN["batch_size"],
        verbose=1,
        tensorboard_log=TRAIN["log_dir"],
        seed=TRAIN["seed"],
    )

    print(f"\n[ppo] Parâmetros: {model.num_timesteps:,} timesteps iniciais")
    print(f"[ppo] Policy: {model.policy}")

    # ------------------------------------------------------------------
    # 6. Callbacks
    # ------------------------------------------------------------------
    model_dir = Path(TRAIN["model_dir"])
    model_dir.mkdir(exist_ok=True)

    # Salva checkpoint a cada 50k passos
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // TRAIN["n_envs"],
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_crypto",
        verbose=1,
    )

    # Avalia no conjunto de validação a cada 20k passos.
    # Salva o modelo que atingiu melhor recompensa média — não o último.
    # Isso é early stopping implícito contra overfitting.
    eval_cb = EvalCallback(
        eval_env=env_val,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=20_000 // TRAIN["n_envs"],
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 7. Treino
    # ------------------------------------------------------------------
    print(f"\n[treino] Iniciando {TRAIN['total_timesteps']:,} timesteps...")
    print("[treino] Acompanhe em tempo real: tensorboard --logdir runs/\n")

    model.learn(
        total_timesteps=TRAIN["total_timesteps"],
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # ------------------------------------------------------------------
    # 8. Salva modelo final
    # ------------------------------------------------------------------
    final_path = model_dir / "ppo_crypto_final"
    model.save(str(final_path))
    print(f"\n[ok] Modelo final salvo em {final_path}.zip")
    print(f"[ok] Melhor modelo (validação) em {model_dir / 'best' / 'best_model.zip'}")


if __name__ == "__main__":
    train()
