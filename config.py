"""
Parâmetros centralizados do projeto.

Alterar aqui reflete em todos os módulos — sem magic numbers espalhados.
"""

ENV = {
    # Janela de observação: quantos candles passados o agente enxerga de uma vez
    "window_size": 48,

    "initial_capital": 1000.0,

    # Custos realistas Binance Futures (taker)
    "fee_rate": 0.0004,        # 0.04% Binance Futures taker (com short)
    "slippage_rate": 0.0003,   # 0.03%

    # Episódio encerra se drawdown atingir este limite
    "max_drawdown_limit": 0.25,

    # Janela rolante para bônus de Sharpe incremental na recompensa
    "sharpe_window": 48,
}

DATA = {
    "exchange": "binance",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "days_train": 730,
    "days_test": 180,
}

TRAIN = {
    "total_timesteps": 300_000,
    "n_envs": 2,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "n_steps": 2048,
    "seed": 42,
    "log_dir": "runs/",
    "model_dir": "models/",

    # Rede menor (32×32 vs 64×64 anterior) — reduz overfitting
    # O problema anterior era rede grande memorizando o bull de 2023-2025
    "policy_kwargs": {"net_arch": [32, 32]},

    # Coeficiente de entropia: mantém o agente explorando por mais tempo
    # Sem isso, a política colapsa para "sempre long" cedo demais
    "ent_coef": 0.05,   # alto: força exploração das 3 ações por mais tempo
}

WALK_FORWARD = {
    # Divide o histórico em N+1 fatias; treina em [0..i], testa em [i+1]
    # Com 910 dias e 4 janelas: ~182 dias por fatia (~6 meses cada)
    "n_windows": 4,
    "timesteps_per_window": 300_000,
    "model_dir": "models/wf/",
    "results_dir": "results/",
}
