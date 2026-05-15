"""
Dashboard de monitoramento do paper trading do agente BTC.
Le o log.csv diretamente do GitHub e atualiza a cada 5 minutos.
Deploy gratuito no Streamlit Cloud.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Configuracao
# ------------------------------------------------------------------

LOG_URL = (
    "https://raw.githubusercontent.com/emerson198729/crypto-agent"
    "/main/paper_trading/log.csv"
)
INITIAL_EQUITY = 1_000.0

st.set_page_config(
    page_title="BTC Agent — Paper Trading",
    page_icon="🤖",
    layout="wide",
)

# ------------------------------------------------------------------
# Carrega dados
# ------------------------------------------------------------------

@st.cache_data(ttl=300)   # cache de 5 minutos
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(LOG_URL, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as exc:
        st.error(f"Erro ao carregar log: {exc}")
        return pd.DataFrame()


df = load_data()

# ------------------------------------------------------------------
# Cabecalho
# ------------------------------------------------------------------

st.title("🤖 BTC Agent — Paper Trading Live")

if df.empty or len(df) < 2:
    st.warning("Aguardando dados... O agente acabou de iniciar. Volte em 1 hora!")
    st.info("O agente toma uma decisao por hora automaticamente via GitHub Actions.")
    st.stop()

ultimo = df["timestamp"].iloc[-1]
st.caption(
    f"Atualizado a cada hora via GitHub Actions  |  "
    f"Ultima decisao: **{ultimo.strftime('%Y-%m-%d %H:%M UTC')}**  |  "
    f"Total de candles: **{len(df)}**"
)

# ------------------------------------------------------------------
# Metricas principais
# ------------------------------------------------------------------

equity_atual    = float(df["equity"].iloc[-1])
retorno_total   = float(df["retorno_acumulado_pct"].iloc[-1])
bh_return       = (df["preco"].iloc[-1] / df["preco"].iloc[0] - 1.0) * 100.0
alpha           = retorno_total - bh_return
n_trades        = int(df["n_trades"].iloc[-1])
preco_atual     = float(df["preco"].iloc[-1])

# Win rate (candles em que havia posicao ativa)
ativos = df[df["posicao_atual"] != 0]
win_rate = float((ativos["retorno_candle_pct"] > 0).mean() * 100) if not ativos.empty else 0.0

# Profit factor
gains  = df[df["retorno_candle_pct"] > 0]["retorno_candle_pct"].sum()
losses = df[df["retorno_candle_pct"] < 0]["retorno_candle_pct"].abs().sum()
pf     = round(gains / losses, 2) if losses > 0 else float("inf")

# Max Drawdown
mdd = float(
    ((df["equity"] - df["equity"].cummax()) / df["equity"].cummax()).min() * 100
)

# Posicao atual
pos_map   = {1: "🟢 LONG", 0: "⚪ FLAT", -1: "🔴 SHORT"}
pos_atual = int(df["posicao_atual"].iloc[-1])
pos_label = pos_map.get(pos_atual, "?")

# Distribuicao de posicoes
pct_long  = (df["posicao_atual"] == 1).mean()  * 100
pct_short = (df["posicao_atual"] == -1).mean() * 100
pct_flat  = (df["posicao_atual"] == 0).mean()  * 100

st.divider()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Posicao atual",   pos_label)
c2.metric("BTC/USD",         f"${preco_atual:,.0f}")
c3.metric("Equity",          f"${equity_atual:,.2f}",
          delta=f"{retorno_total:+.2f}%")
c4.metric("Alpha vs B&H",    f"{alpha:+.2f} pp",
          delta=f"B&H: {bh_return:+.2f}%",
          delta_color="off")
c5.metric("Trades totais",   n_trades)
c6.metric("Max Drawdown",    f"{mdd:.2f}%",
          delta_color="inverse")

c7, c8, c9 = st.columns(3)
c7.metric("Win Rate",        f"{win_rate:.1f}%")
c8.metric("Profit Factor",   f"{pf}" if pf != float('inf') else "∞")
c9.metric("Long / Flat / Short",
          f"{pct_long:.0f}% / {pct_flat:.0f}% / {pct_short:.0f}%")

st.divider()

# ------------------------------------------------------------------
# Grafico 1: Curva de equity vs Buy & Hold
# ------------------------------------------------------------------

df["equity_norm"] = df["equity"] / INITIAL_EQUITY * 100.0
df["bh_norm"]     = df["preco"]  / float(df["preco"].iloc[0]) * 100.0

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=df["timestamp"], y=df["equity_norm"],
    name="Agente RL",
    line=dict(color="#00C4FF", width=2),
    hovertemplate="<b>Agente</b>: %{y:.2f}<br>%{x}<extra></extra>",
))

fig1.add_trace(go.Scatter(
    x=df["timestamp"], y=df["bh_norm"],
    name="Buy & Hold BTC",
    line=dict(color="#FF9500", width=2, dash="dot"),
    hovertemplate="<b>B&H</b>: %{y:.2f}<br>%{x}<extra></extra>",
))

fig1.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4)

fig1.update_layout(
    title="Curva de Equity — Agente vs Buy & Hold (base 100)",
    xaxis_title="Data",
    yaxis_title="Valor (base 100)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
    template="plotly_dark",
    hovermode="x unified",
)

st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------------
# Grafico 2: Posicoes ao longo do tempo + preco BTC
# ------------------------------------------------------------------

fig2 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.04,
    subplot_titles=("Preco BTC/USD", "Posicao do Agente"),
)

# Preco BTC
fig2.add_trace(go.Scatter(
    x=df["timestamp"], y=df["preco"],
    name="BTC/USD",
    line=dict(color="#FF9500", width=1.5),
    hovertemplate="$%{y:,.0f}<extra>BTC/USD</extra>",
), row=1, col=1)

# Faixas coloridas de posicao (LONG = verde, SHORT = vermelho)
for _, row in df.iterrows():
    if row["posicao_atual"] == 1:
        color, opacity = "rgba(0,200,100,0.15)", 0.8
    elif row["posicao_atual"] == -1:
        color, opacity = "rgba(255,80,80,0.15)", 0.8
    else:
        continue
    fig2.add_vrect(
        x0=row["timestamp"],
        x1=row["timestamp"] + pd.Timedelta(hours=1),
        fillcolor=color, opacity=opacity, line_width=0,
        row=1, col=1,
    )

# Barra de posicao (1 / 0 / -1)
cores = df["posicao_atual"].map({1: "#00C878", 0: "#888888", -1: "#FF5050"})
fig2.add_trace(go.Bar(
    x=df["timestamp"], y=df["posicao_atual"],
    name="Posicao",
    marker_color=cores,
    hovertemplate="%{x}<br>Posicao: %{y}<extra></extra>",
), row=2, col=1)

fig2.update_layout(
    height=500,
    template="plotly_dark",
    showlegend=False,
    hovermode="x unified",
)
fig2.update_yaxes(title_text="USD", row=1, col=1)
fig2.update_yaxes(
    title_text="Posicao",
    tickvals=[-1, 0, 1],
    ticktext=["SHORT", "FLAT", "LONG"],
    row=2, col=1,
)

st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# Grafico 3: Retorno por candle (barras)
# ------------------------------------------------------------------

cores_ret = df["retorno_candle_pct"].apply(
    lambda v: "#00C878" if v > 0 else ("#FF5050" if v < 0 else "#888888")
)

fig3 = go.Figure(go.Bar(
    x=df["timestamp"],
    y=df["retorno_candle_pct"],
    marker_color=cores_ret,
    hovertemplate="%{x}<br>Retorno: %{y:.3f}%<extra></extra>",
))
fig3.add_hline(y=0, line_color="white", line_width=0.5, opacity=0.3)
fig3.update_layout(
    title="Retorno por Candle (%)",
    height=280,
    template="plotly_dark",
    yaxis_title="%",
)
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------
# Tabela: ultimas decisoes
# ------------------------------------------------------------------

st.subheader("Ultimas decisoes do agente")

pos_emoji = {1: "🟢 LONG", 0: "⚪ FLAT", -1: "🔴 SHORT"}
df_show = df.tail(20).copy()[::-1].reset_index(drop=True)
df_show["timestamp"]       = df_show["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
df_show["posicao_atual"]   = df_show["posicao_atual"].map(pos_emoji)
df_show["posicao_anterior"]= df_show["posicao_anterior"].map(pos_emoji)
df_show["preco"]           = df_show["preco"].apply(lambda v: f"${v:,.2f}")
df_show["equity"]          = df_show["equity"].apply(lambda v: f"${v:,.2f}")
df_show["retorno_candle_pct"] = df_show["retorno_candle_pct"].apply(
    lambda v: f"{v:+.3f}%"
)
df_show["retorno_acumulado_pct"] = df_show["retorno_acumulado_pct"].apply(
    lambda v: f"{v:+.2f}%"
)

st.dataframe(
    df_show[[
        "timestamp", "preco", "posicao_anterior", "decisao",
        "posicao_atual", "equity", "retorno_candle_pct", "retorno_acumulado_pct"
    ]].rename(columns={
        "timestamp":            "Hora (UTC)",
        "preco":                "BTC/USD",
        "posicao_anterior":     "Posicao Anterior",
        "decisao":              "Decisao",
        "posicao_atual":        "Posicao Nova",
        "equity":               "Equity",
        "retorno_candle_pct":   "Retorno Candle",
        "retorno_acumulado_pct":"Retorno Acum.",
    }),
    use_container_width=True,
    hide_index=True,
)

# ------------------------------------------------------------------
# Rodape
# ------------------------------------------------------------------

st.divider()
st.caption(
    "Agente RL treinado com PPO (Stable Baselines3) | "
    "Features: ret_1/4/24/168/720, RSI, Bollinger, ATR, Volume, MACD | "
    "Dados: Kraken BTC/USD | "
    "Infraestrutura: GitHub Actions (gratuito)"
)
