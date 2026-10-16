"""Small HTML chart helpers (bars and heatmaps) shared by the dashboard pages."""

from __future__ import annotations

import html

MODEL_COLORS = {"Random Forest": "#1DB954", "K-Nearest Neighbors": "#d0d0d0", "Logistic Regression": "#7a7a7a"}


def legend(colors: dict) -> str:
    return "<div class='lg'>" + "".join(
        f"<span><i style='background:{c}'></i>{html.escape(n)}</span>" for n, c in colors.items()) + "</div>"


def gbars(rows: list[dict], colors: dict, vmax: float, fmt=None) -> str:
    """Grouped horizontal bars. rows: [{"l": label, "v": {series: value}, "s": {series: sd}}]."""
    out = []
    for r in rows:
        bars = ""
        for name, color in colors.items():
            v = r["v"].get(name)
            if v is None:
                continue
            sd = (r.get("s") or {}).get(name) or 0
            whisker = (f"<span class='wk' style='left:{max(v - sd, 0) / vmax * 100:.2f}%;width:{2 * sd / vmax * 100:.2f}%'></span>"
                       if sd else "")
            text = fmt(v, sd) if fmt else f"{v:.3f}"
            bars += (f"<div class='bw'><div class='tr'><div class='b' style='width:{max(v / vmax * 100, .6):.2f}%;"
                     f"background:{color}'></div>{whisker}</div><span class='bv'>{text}</span></div>")
        out.append(f"<div class='gr'><div class='gl'>{html.escape(r['l'])}</div><div class='gb'>{bars}</div></div>")
    return "".join(out)


def heatmap(conf, genres: list[str], support, title: str, color: str) -> str:
    """Confusion matrix as a CSS grid. Brightness = share of that true genre's test tracks."""
    # Cell brightness is the share of that true genre's test tracks, so rows are comparable even if their sizes differ.
    n = len(genres)
    cells = ["<div></div>"] + [f"<div class='hc'>{html.escape(g[:13])}</div>" for g in genres]
    diag = 0
    for i in range(n):
        cells.append(f"<div class='hr'>{html.escape(genres[i][:15])}</div>")
        diag += conf[i][i]
        for j in range(n):
            share = conf[i][j] / support[i] if support[i] else 0
            alpha = min(1, .12 + share ** .5 * 1.1) if i == j else min(.85, share ** .5 * 1.4)
            rgb = "29,185,84" if i == j else "232,232,232"
            tip = html.escape(f"True {genres[i]} → predicted {genres[j]}: {int(conf[i][j])} tracks ({share:.0%})")
            cells.append(f"<div class='hx' title='{tip}' style='background:rgba({rgb},{alpha:.2f})'></div>")
    total = sum(support) or 1
    return (f"<figure><div class='hg' style='grid-template-columns:96px repeat({n},1fr)'>{''.join(cells)}</div>"
            f"<figcaption><b>{html.escape(title)}</b><br>{diag / total:.0%} of these genres' test tracks "
            "are on the diagonal (correct)</figcaption></figure>")
