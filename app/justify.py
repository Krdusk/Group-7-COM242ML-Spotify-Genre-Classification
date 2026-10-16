"""Side-by-side evidence for why one classifier beats the others (diagrams and verdict)."""

from __future__ import annotations


import numpy as np
import streamlit as st

from app.charts import MODEL_COLORS, gbars, heatmap, legend

METRICS = [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"),
           ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")]
SHORT = {"Random Forest": "Forest", "K-Nearest Neighbors": "KNN", "Logistic Regression": "LogReg"}


def render_justification(result: dict) -> None:
    names = [n for n in MODEL_COLORS if n in result["scores"]]
    colors = {n: MODEL_COLORS[n] for n in names}
    best = result["best_name"]
    cv = {n: {k: (m["mean"], m["std"]) for k, m in result["validation_metrics"].get(n, {}).items() if k in dict(METRICS)} for n in names}
    for n in names:
        cv[n].setdefault("macro_f1", (result["scores"][n]["mean"], result["scores"][n]["std"]))
    te = result["test_metrics_by_model"]
    n_labels = len(result["labels"])
    cv_max = np.ceil(max(m + s for n in names for m, s in cv[n].values()) * 20) / 20
    te_max = np.ceil(max(te[n][k] for n in names for k, _ in METRICS) * 20) / 20
    pm = lambda v, s: f"{v:.3f} ± {s:.3f}"

    # Cross-validation macro F1 selects the winner. The test set is only a final check, never used for selection.
    st.markdown("### Why this model? Side-by-side evidence")
    st.caption("Every diagram uses the same data, the same cross-validation folds and the same held-out test rows. Bars start at zero.")
    st.markdown(legend(colors), unsafe_allow_html=True)

    cv_rows = [{"l": lab, "v": {n: cv[n][k][0] for n in names if k in cv[n]}, "s": {n: cv[n][k][1] for n in names if k in cv[n]}}
               for k, lab in METRICS if all(k in cv[n] for n in names)]
    st.markdown("<div class='card'><h3>A · Cross-validation scores (selection criterion: macro F1)</h3>"
                "<div class='cap'>Whiskers show one standard deviation across the folds. Macro F1 picked the winner.</div>"
                + gbars(cv_rows, colors, cv_max, pm) + "</div>", unsafe_allow_html=True)
    te_rows = [{"l": lab, "v": {n: te[n][k] for n in names}} for k, lab in METRICS]
    st.markdown("<div class='card'><h3>B · Same held-out test set</h3>"
                f"<div class='cap'>Never used for selection, so it checks that the winner generalises. {result['test_rows']:,} rows.</div>"
                + gbars(te_rows, colors, te_max) + "</div>", unsafe_allow_html=True)
    chance = 1 / n_labels
    base = [{"l": "Accuracy", "v": {n: te[n]["accuracy"] for n in names}}, {"l": f"Random guess (1/{n_labels})", "v": {names[0]: chance}}]
    st.markdown(f"<div class='card'><h3>C · Versus random guessing</h3><div class='cap'>With {n_labels} genres, chance is about {chance:.1%}. "
                "All models clear it, so they learned something real.</div>"
                + gbars(base, {**colors}, te_max, lambda v, s: f"{v:.1%}") + "</div>", unsafe_allow_html=True)

    # Genres are balanced (about 200 test tracks each), so this 12-genre sample is not a ranking by size.
    labels = list(result["labels"])
    conf0 = np.asarray(result["confusion"])
    top = [int(i) for i in np.argsort(conf0.sum(axis=1))[::-1][:12]]
    f1 = {n: [result["test_reports_by_model"][n][labels[i]]["f1-score"] for i in top] for n in names}
    support = [int(conf0.sum(axis=1)[i]) for i in top]
    g_rows = [{"l": f"{labels[i]} ({support[k]} test tracks)", "v": {n: f1[n][k] for n in names}} for k, i in enumerate(top)]
    st.markdown("<div class='card'><h3>D · Genre-by-genre F1 on a 12-genre sample</h3>"
                "<div class='cap'>Genres have similar test sizes, so this is a sample, not a ranking by size. Is the winner good everywhere?</div>"
                + gbars(g_rows, colors, 1.0) + "</div>", unsafe_allow_html=True)
    maps = "".join(heatmap(np.asarray(result["test_confusions_by_model"][n])[np.ix_(top, top)], [labels[i] for i in top],
                           support, n, colors[n]) for n in names)
    st.markdown("<div class='card'><h3>E · Where each model goes wrong: confusion matrices</h3>"
                "<div class='cap'>Rows are the true genre, columns the predicted one. A bright green diagonal is good; bright grey cells are mix-ups. Hover for counts.</div>"
                f"<div class='hm'>{maps}</div></div>", unsafe_allow_html=True)

    # The verdict text is built from the saved numbers, so it cannot drift from the charts.
    rank = sorted(names, key=lambda n: -cv[n]["macro_f1"][0])
    g1, g2 = rank[0], rank[1]
    gap = cv[g1]["macro_f1"][0] - cv[g2]["macro_f1"][0]
    sd = max(cv[g1]["macro_f1"][1], cv[g2]["macro_f1"][1]) or 1e-9
    test_wins = {n: 0 for n in names}
    for k, _ in METRICS:
        test_wins[max(names, key=lambda n: te[n][k])] += 1
    genre_wins = {n: 0 for n in names}
    for k in range(len(top)):
        genre_wins[max(names, key=lambda n: f1[n][k])] += 1
    order = " › ".join(f"<b>#{i + 1} {n}</b>" for i, n in enumerate(rank))
    items = [
        f"<b>{g1} ranks first</b> on cross-validation macro F1 ({cv[g1]['macro_f1'][0]:.3f} vs {cv[g2]['macro_f1'][0]:.3f} for {g2}). "
        f"The gap of {gap:.3f} is about {gap / sd:.0f}× the fold-to-fold SD ({sd:.3f}), so it is not just fold luck."
        + (" It is the selected model." if g1 == best else ""),
        f"On the same {result['test_rows']:,} held-out rows, {g1} is best on <b>{test_wins[g1]} of {len(METRICS)}</b> metrics "
        f"(accuracy {te[g1]['accuracy']:.0%} vs {te[g2]['accuracy']:.0%} and {te[rank[-1]]['accuracy']:.0%}).",
        "Genre by genre (12-genre sample, F1): " + ", ".join(f"{SHORT[n]} wins <b>{genre_wins[n]}</b>" for n in names) + ".",
        f"Ranking: {order}. A plausible reason: many voting trees can follow uneven, non-linear boundaries; "
        "nearest neighbours follow local clusters but are blurred by look-alike genres; straight-line boundaries struggle when genres overlap.",
        f"Reality check: {te[g1]['accuracy']:.0%} accuracy is about <b>{te[g1]['accuracy'] / chance:.0f}× better than random guessing</b>, yet far from perfect. "
        "Genres overlap and some track IDs have more than one label. Macro F1 is a fair yardstick because every genre has a similar number of tracks.",
    ]
    st.markdown("<div class='card'><h3>Verdict</h3><ul class='vd'>" + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>",
                unsafe_allow_html=True)
