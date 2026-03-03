"""QUBO シミュレーテッドアニーリング — エントリーポイント。"""

import streamlit as st

st.set_page_config(
    page_title="QUBO SA シミュレータ",
    page_icon="🌡️",
    layout="wide",
)

# ── ページ定義 ──────────────────────────
number_partitioning = st.Page(
    "pages/number_partitioning/page.py",
    title="数分割問題",
    icon="✂️",
    url_path="number_partitioning",
)
recommendation = st.Page(
    "pages/recommendation/page.py",
    title="レコメンドシステム",
    icon="🛍️",
    url_path="recommendation",
)


# ── ホームページ本体 ─────────────────────
def home() -> None:
    st.title("🌡️ QUBO シミュレーテッドアニーリング")
    st.markdown(
        """
シミュレーテッドアニーリング (SA) を使って、さまざまな組み合わせ最適化問題を  
QUBO にエンコードしてブラウザ上で解くシミュレータ集です。

問題カードをクリックしてシミュレータに移動してください。
"""
    )
    st.divider()
    st.subheader("📋 実装済み問題")

    PROBLEMS = [
        {
            "name": "✂️ 数分割問題",
            "desc": "数列を 2 グループに等分割する組み合わせを探索します。",
            "qubo": "minimize (Σ (2xᵢ − 1) nᵢ)²",
            "page": number_partitioning,
            "status": "✅ 実装済み",
        },
        {
            "name": "🛍️ レコメンドシステム",
            "desc": "21 商品カタログから必須カテゴリ・予算制約を満たす最適カートを SA で計算します。",
            "qubo": "minimize λ_req·Req + λ_opt·Opt + λ_budget·Budget − λ_score·Score",
            "page": recommendation,
            "status": "✅ 実装済み",
        },
    ]

    for p in PROBLEMS:
        with st.container(border=True):
            col_title, col_status = st.columns([4, 1])
            col_title.markdown(f"### {p['name']}")
            col_status.markdown(f"<br>{p['status']}", unsafe_allow_html=True)
            st.markdown(p["desc"])
            st.markdown(f"**QUBO 目的関数**: `{p['qubo']}`")
            st.page_link(p["page"], label="▶ シミュレータを開く", icon="🔗")

    st.divider()
    st.caption("今後追加予定の問題: 最大カット問題、グラフ彩色問題、巡回セールスマン問題 など")


# ── ナビゲーション登録・実行 ─────────────
pg = st.navigation(
    {
        "ホーム": [st.Page(home, title="ホーム", icon="🌡️", default=True)],
        "問題": [number_partitioning, recommendation],
    }
)
pg.run()
