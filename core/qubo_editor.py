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


_DEFAULT_DESCRIPTION = """\
`PARAMS` 辞書にパラメータを追加すると、スライダーが**自動で更新**されます。  
`build_qubo` 関数を書き換えて定式化を変更してください。  
編集後は **Ctrl+Enter** またはエディタ外クリックで自動適用されます。
"""


def qubo_editor(
    default_code: str,
    session_prefix: str,
    description: str | None = None,
) -> tuple[dict, object]:
    """
    QUBO コードエディタを描画し、(PARAMS, build_qubo_fn) を返す。

    Parameters
    ----------
    default_code    : ページ固有のデフォルト QUBO コード文字列
    session_prefix  : セッションステートのキー衝突を防ぐプレフィックス
                      (ページごとに一意な文字列を指定)
    description     : エディタ上部に表示する説明文 (None でデフォルト文)

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
        st.markdown(description if description is not None else _DEFAULT_DESCRIPTION)
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


def render_qubo_params(
    params_spec: dict,
    key_prefix: str,
    n_cols: int = 2,
) -> dict:
    """
    PARAMS 仕様辞書からスライダーを自動生成し、現在値を返す。

    Parameters
    ----------
    params_spec : PARAMS 辞書 (qubo_editor が返すもの)
    key_prefix  : ウィジェットキーのプレフィックス (ページごとに一意)
    n_cols      : 列数 (default 2)

    Returns
    -------
    dict : { param_key: 現在値, ... }
    """
    values: dict = {}
    if not params_spec:
        return values
    cols = st.columns(min(n_cols, len(params_spec)))
    for idx, (key, spec) in enumerate(params_spec.items()):
        label   = spec.get("label",   key)
        ptype   = spec.get("type",    "float")
        default = spec.get("default", 1.0)
        pmin    = spec.get("min",     0.0)
        pmax    = spec.get("max",     10.0)
        step    = spec.get("step",    0.1 if ptype == "float" else 1)
        col = cols[idx % len(cols)]
        with col:
            if ptype == "int":
                values[key] = st.slider(
                    label,
                    min_value=int(pmin), max_value=int(pmax),
                    value=int(default), step=int(step),
                    key=f"{key_prefix}__{key}",
                )
            elif ptype == "float":
                values[key] = st.slider(
                    label,
                    min_value=float(pmin), max_value=float(pmax),
                    value=float(default), step=float(step),
                    key=f"{key_prefix}__{key}",
                )
            else:
                values[key] = st.text_input(
                    label, value=str(default), key=f"{key_prefix}__{key}",
                )
    return values
