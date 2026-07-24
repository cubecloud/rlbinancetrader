"""Статистика для корзин action_space (запрос пользователя 23.07).

Считает по сделкам v7 и минутным OHLC:
1. Для головы 3 (лимитки): глубина выброса ВНИЗ после решения о входе —
   насколько ниже цены исполнения (open следующего бара) проваливается low
   в течение TTL-окна. Даёт вероятность исполнения limit-buy по смещениям.
   Аналогично для выхода: выброс ВВЕРХ после решения о выходе (limit-sell).
2. Для головы 4 (SL): MAE — максимальная просадка внутри сделки от цены
   входа; сколько сделок проваливается глубже порога и чем они кончаются.
3. Для головы 5 (TP): MFE — максимальный взлёт внутри сделки; сколько
   сделок достигает порога.

Run: python strategy_rl/stats_action_bins.py <state.parquet> <trades_labeled.csv>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

TTLS = (30, 60)                      # минуты жизни лимитки
LIMIT_GRID = (0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.00)   # % смещения
SL_GRID = (2.0, 3.5, 4.5, 5.0, 6.0, 7.0)                  # % просадки
TP_GRID = (3.0, 5.0, 6.0, 8.0, 9.0, 12.0, 15.0)           # % взлёта


def main():
    st = pd.read_parquet(sys.argv[1], columns=["open", "high", "low", "close"])
    tr = pd.read_csv(sys.argv[2])
    o = st["open"].to_numpy(float)
    h = st["high"].to_numpy(float)
    lo = st["low"].to_numpy(float)
    n = len(st)

    # --- 1а. Вход: выброс вниз после решения (limit-buy fill probability) ---
    print("=== ГОЛОВА 3: выбросы ВНИЗ после решения о входе ===")
    for ttl in TTLS:
        dips = []
        for eb in tr["entry_bar"].astype(int):
            if eb + 1 >= n:
                continue
            fill = o[eb]                       # market-цена v7 (open бара входа)
            w_lo = lo[eb: min(eb + ttl, n)]
            dips.append((fill - w_lo.min()) / fill * 100.0)
        dips = np.array(dips)
        pct = {f"p{p}": round(float(np.percentile(dips, p)), 3)
               for p in (25, 50, 75, 90)}
        print(f"TTL {ttl}м: глубина макс. провала ниже цены входа, %: {pct}")
        fills = {f"{g}%": f"{float((dips >= g).mean()):.0%}" for g in LIMIT_GRID}
        print(f"  вероятность исполнения limit-buy по смещению: {fills}")

    # --- 1б. Выход: выброс вверх после решения о выходе (limit-sell) ---
    print("=== ГОЛОВА 3: выбросы ВВЕРХ после решения о выходе (без SL/forced) ===")
    dec_exits = tr[~tr["exit_reason"].isin(("sl", "forced_eod"))]
    for ttl in TTLS:
        rises = []
        for xb in dec_exits["exit_bar"].astype(int):
            if xb + 1 >= n:
                continue
            fill = o[xb]
            w_hi = h[xb: min(xb + ttl, n)]
            rises.append((w_hi.max() - fill) / fill * 100.0)
        rises = np.array(rises)
        pct = {f"p{p}": round(float(np.percentile(rises, p)), 3)
               for p in (25, 50, 75, 90)}
        print(f"TTL {ttl}м: высота макс. взлёта над ценой выхода, %: {pct}")
        fills = {f"{g}%": f"{float((rises >= g).mean()):.0%}" for g in LIMIT_GRID}
        print(f"  вероятность исполнения limit-sell по смещению: {fills}")

    # --- 2. MAE (для SL) и 3. MFE (для TP) ---
    mae, mfe, rets = [], [], []
    for t in tr.itertuples():
        eb, xb = int(t.entry_bar), min(int(t.exit_bar), n - 1)
        ep = float(t.entry_price)
        seg_lo = lo[eb: xb + 1]
        seg_hi = h[eb: xb + 1]
        mae.append((ep - seg_lo.min()) / ep * 100.0)
        mfe.append((seg_hi.max() - ep) / ep * 100.0)
        rets.append(float(t.return_pct) * 100.0)
    mae, mfe, rets = np.array(mae), np.array(mfe), np.array(rets)

    print("=== ГОЛОВА 4: MAE (макс. просадка внутри сделки от входа) ===")
    print({f"p{p}": round(float(np.percentile(mae, p)), 2) for p in (50, 75, 90, 95)})
    for g in SL_GRID:
        hit = mae >= g
        if hit.any():
            rec = rets[hit]
            print(f"  просадка ≥{g}%: {hit.mean():.0%} сделок; из них закрылись "
                  f"в плюс {(rec > 0).mean():.0%}, средний итог {rec.mean():+.2f}%")

    print("=== ГОЛОВА 5: MFE (макс. взлёт внутри сделки) ===")
    print({f"p{p}": round(float(np.percentile(mfe, p)), 2) for p in (25, 50, 75, 90)})
    for g in TP_GRID:
        hit = mfe >= g
        print(f"  взлёт ≥{g}%: {hit.mean():.0%} сделок"
              + (f"; их средний фактический итог {rets[hit].mean():+.2f}%"
                 if hit.any() else ""))


if __name__ == "__main__":
    main()
