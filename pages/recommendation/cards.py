"""レコメンドシステム — 共通商品カード HTML。"""

from __future__ import annotations

from pages.recommendation.items_data import CATEGORY_BG, CATEGORY_COLOR, Item


def item_card_html(item: Item, selected: bool = False) -> str:
    """入力・出力両方で使う統一商品カード。

    Parameters
    ----------
    item:
        表示する商品。
    selected:
        True のとき「選択中」バッジを表示し、ボーダーを強調する。
    """
    cat_color = CATEGORY_COLOR.get(item.category, "#888")
    cat_bg = CATEGORY_BG.get(item.category, "#f0f0f0")
    border = f"2px solid {cat_color}" if selected else f"1px solid {cat_color}66"
    shadow = "0 4px 16px rgba(0,0,0,0.12)" if selected else "0 1px 4px rgba(0,0,0,0.07)"
    badge_html = (
        '<span style="position:absolute;top:8px;right:8px;'
        'background:#4CAF50;color:white;font-size:11px;'
        'padding:2px 8px;border-radius:12px;font-weight:bold;">✓ 選択中</span>'
        if selected else ""
    )

    return f"""
<div style="
  position:relative;
  border:{border};
  border-radius:14px;
  background:linear-gradient(135deg,#ffffff 0%,{cat_bg} 100%);
  padding:16px 14px 12px;
  box-shadow:{shadow};
  height:100%;
">
  {badge_html}

  <!-- アイコン -->
  <div style="font-size:42px;text-align:center;margin-bottom:6px;">
    {item.emoji}
  </div>

  <!-- 商品名 -->
  <div style="font-weight:800;font-size:14px;text-align:center;color:#1a1a2e;margin-bottom:6px;line-height:1.3;">
    {item.name}
  </div>

  <!-- カテゴリバッジ -->
  <div style="text-align:center;margin-bottom:8px;">
    <span style="
      background:{cat_bg};color:{cat_color};font-size:11px;
      padding:2px 10px;border-radius:10px;font-weight:700;
      border:1px solid {cat_color}40;
    ">{item.category}</span>
  </div>

  <!-- 説明 -->
  <div style="font-size:12px;color:#555;text-align:center;margin-bottom:10px;line-height:1.5;">
    {item.description}
  </div>

  <!-- 評価バー -->
  <div style="margin-bottom:8px;">
    <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px;">
      <span style="color:#f59f00;font-weight:600;">★ {item.score:.1f}/5.0</span>
      <span style="color:#aaa;">評価スコア</span>
    </div>
    <div style="background:#eee;border-radius:4px;height:5px;">
      <div style="background:linear-gradient(90deg,#f59f00,#ff6b35);border-radius:4px;height:5px;width:{item.score/5*100:.0f}%;"></div>
    </div>
  </div>

  <!-- 価格 -->
  <div style="
    background:linear-gradient(90deg,{cat_color}22,{cat_color}11);
    border-radius:8px;padding:7px 10px;
    text-align:right;margin-top:4px;
  ">
    <span style="font-size:17px;font-weight:800;color:{cat_color};">¥{item.price:,}</span>
  </div>
</div>
"""


def compact_card_html(item: Item) -> str:
    """非推奨商品向けコンパクト横並びカード。"""
    return f"""
<div style="
  border:1px solid #ddd;border-radius:10px;
  background:#f8f8f8;padding:10px 12px;
  display:flex;align-items:center;gap:10px;
  opacity:0.65;
">
  <span style="font-size:22px;">{item.emoji}</span>
  <div style="flex:1;min-width:0;">
    <div style="font-size:13px;font-weight:600;color:#555;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
      {item.name}
    </div>
    <div style="font-size:11px;color:#888;">¥{item.price:,} &nbsp;★{item.score:.1f}</div>
  </div>
  <span style="font-size:11px;color:#aaa;white-space:nowrap;">非選択</span>
</div>
"""
