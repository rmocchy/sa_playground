"""共通 QUBO コードエディタコンポーネント。

各ページから以下のように呼び出す:

    from core.qubo_editor import qubo_editor

    PARAMS, build_qubo_fn = qubo_editor(
        default_code=DEFAULT_QUBO_CODE,
        session_prefix="number_partitioning",  # ページごとに一意な文字列
    )
"""

from __future__ import annotations

import traceback

import streamlit as st
from streamlit_ace import st_ace


def qubo_editor(
    default_code: str,
    session_prefix: str,
) -> tuple[dict, object]:
    """
    QUBO コードエディタを描画し、(PARAMS, build_qubo_fn) を返す。

    Parameters
    ----------
    default_code    : ページ固有のデフォルト QUBO コード文字列
    session_prefix  : セッションステートのキー衝突を防ぐプレフィックス
                      (ページごとに一意な文字列を指定)

    Returns
    -------
    PARAMS        : QUBO コード内で定義された PARAMS 辞書
    build_qubo_fn : QUBO コード内で定義された build_qubo 関数

    エラー時は st.error を表示して st.stop() を呼ぶ。
    """
    key_code = f"{session_prefix}__qubo_code"
    key_gen = f"{session_prefix}__editor_gen"
    key_applied = f"{session_prefix}__qubo_applied"

    if key_gen not in st.session_state:
        st.session_state[key_gen] = 0

    with st.expander("🖊️ QUBO 定式化コードを編集", expanded=False):
        st.markdown(
            """
`PARAMS` 辞書にパラメータを追加すると、サイドバーのコントロールが**自動で更新**されます。  
`build_qubo(numbers, params)` 関数を書き換えて定式化を変更してください。  
編集後は **Ctrl+Enter** またはエディタ内の Apply ボタンで適用されます。
"""
        )
        edited_code: str = st_ace(
            value=st.session_state.get(key_code, default_code),
            language="python",
            theme="monokai",
            height=420,
            font_size=14,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            key=f"{session_prefix}__ace_{st.session_state[key_gen]}",
        ) or default_code

        if st.button("🔄 デフォルトに戻す", use_container_width=True, key=f"{session_prefix}__reset_btn"):
            st.session_state[key_code] = default_code
            st.session_state[key_gen] += 1
            st.session_state[key_applied] = False
            st.rerun()

    # Apply 検出 (値が変化したら自動適用)
    _code_changed = edited_code != st.session_state.get(key_code, default_code)
    if _code_changed:
        st.session_state[key_code] = edited_code
        st.session_state[key_applied] = False

    current_code: str = st.session_state.get(key_code, default_code)

    # exec
    exec_ns: dict = {}
    code_error: str | None = None
    try:
        exec(compile(current_code, "<qubo_editor>", "exec"), exec_ns)  # noqa: S102
        if "PARAMS" not in exec_ns:
            code_error = "PARAMS 辞書が定義されていません。"
        elif "build_qubo" not in exec_ns:
            code_error = "build_qubo 関数が定義されていません。"
    except Exception:
        code_error = traceback.format_exc()

    if code_error:
        st.error(f"**コードエラー:**\n```\n{code_error}\n```")
        st.stop()

    # exec 成功後に適用メッセージ
    if _code_changed:
        st.session_state[key_applied] = True
    if st.session_state.get(key_applied):
        st.success("✅ QUBO コードを適用しました。")

    return exec_ns["PARAMS"], exec_ns["build_qubo"]
