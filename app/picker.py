"""Pick-a-track experience: search a song, listen, then see what the model predicts and why."""

from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from app.charts import MODEL_COLORS, gbars, legend

FEATURES = [("danceability", "Dance"), ("energy", "Energy"), ("valence", "Mood"), ("acousticness", "Acoustic"),
            ("speechiness", "Speech"), ("instrumentalness", "Instrumental"), ("liveness", "Live")]
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ID_RE = re.compile(r"[A-Za-z0-9]{22}")

PLAYER_HTML = """<!doctype html><html><body style="margin:0;background:transparent;font:13px sans-serif;color:#b3b3b3">
<div id="p"></div><div id="n" style="padding:6px 4px"></div>
<script>
var ID="__ID__",AUTO=__AUTO__,done=false;
function plain(){if(done)return;done=true;document.getElementById('p').innerHTML='<iframe style="border-radius:16px;border:0" src="https://open.spotify.com/embed/track/'+ID+'?utm_source=generator&theme=0" width="100%" height="152" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>';
 document.getElementById('n').textContent='Press play if it does not start by itself.'}
window.onSpotifyIframeApiReady=function(API){done=true;
 API.createController(document.getElementById('p'),{uri:'spotify:track:'+ID,width:'100%',height:152,theme:0},function(c){
  c.addListener('ready',function(){if(AUTO){c.play();document.getElementById('n').textContent='Playing on Spotify. If your browser blocks autoplay, press play. Full songs need you to be logged in to Spotify in this browser; otherwise you hear a preview.'}})})};
var s=document.createElement('script');s.src='https://open.spotify.com/embed/iframe-api/v1';s.async=true;s.onerror=plain;document.head.appendChild(s);
setTimeout(plain,6000);
</script></body></html>"""


def _catalog(result: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """One row per track ID, with search text and a held-out flag (built once)."""
    # A song is "unseen" only if its ID is in the held-out test set. The split is grouped by track ID, so none of its rows were used for training.
    sig = (len(result["X"]), result["best_name"])
    if st.session_state.get("_cat_sig") == sig:
        return st.session_state["_cat"]
    X = result["X"]
    meta = raw.loc[X.index, ["track_id", "artists", "album_name", "track_name"]].copy()
    meta["popularity"] = X["popularity"] if "popularity" in X else 0
    meta["row"] = X.index
    test_ids = set(meta["track_id"].iloc[result["test_indices"]])
    st.session_state["_ids"] = meta["track_id"]
    meta = meta.drop_duplicates("track_id")
    meta["track_name"] = meta["track_name"].fillna("Unknown title")
    meta["artists"] = meta["artists"].fillna("Unknown artist")
    meta["_s"] = (meta["track_name"] + " " + meta["artists"].str.replace(";", " ")).str.lower()
    meta["heldout"] = meta["track_id"].isin(test_ids)
    st.session_state["_cat"], st.session_state["_cat_sig"] = meta, sig
    return meta


def _genre_means(result: dict) -> pd.DataFrame:
    if "_gm" not in st.session_state:
        cols = [c for c, _ in FEATURES if c in result["X"]]
        st.session_state["_gm"] = result["X"][cols].groupby(result["y"]).mean()
    return st.session_state["_gm"]


def _pick(track_id: str) -> None:
    st.session_state.update(picked=track_id, play=None)


def _listen(track_id: str) -> None:
    st.session_state["play"] = track_id


def _stop() -> None:
    st.session_state["play"] = None


def _search(pool: pd.DataFrame, text: str, n: int = 8) -> pd.DataFrame:
    mask = pd.Series(True, index=pool.index)
    for word in text.lower().split():
        mask &= pool["_s"].str.contains(word, regex=False)
    return pool[mask].sort_values("popularity", ascending=False).head(n)


def _bubble_buttons(df: pd.DataFrame, prefix: str) -> None:
    cols = st.columns(2)
    for i, row in enumerate(df.itertuples()):
        label = f"{row.track_name[:34]} · {row.artists.replace(';', ', ')[:26]}"
        cols[i % 2].button(label, key=f"{prefix}_{row.track_id}", on_click=_pick, args=(row.track_id,), use_container_width=True)


def _player_block(track_id: str) -> None:
    """Listen button. Clicking it renders Spotify's player and starts playback."""
    # The ID goes into HTML below, so it must be exactly 22 letters or digits.
    if not ID_RE.fullmatch(track_id):
        return
    if st.session_state.get("play") == track_id:
        st.iframe(PLAYER_HTML.replace("__ID__", track_id).replace("__AUTO__", "true"), height=190)
        st.button("⏹ Close player", key="stop", on_click=_stop)
    else:
        st.button("▶ Listen on Spotify", key="listen", on_click=_listen, args=(track_id,))
        st.caption("Plays right here and starts automatically. Full songs need a Spotify login in this browser; otherwise you hear a preview.")


def _feature_bubbles(row: pd.Series) -> str:
    out = []
    for i, (col, label) in enumerate(FEATURES):
        if col not in row:
            continue
        v = float(row[col])
        size = int(58 + 86 * min(max(v, 0), 1))
        out.append(f"<div class='fb' style='width:{size}px;height:{size}px;background:rgba(29,185,84,{.12 + .5 * v:.2f});"
                   f"animation-delay:{i * .06:.2f}s'><div><b>{v:.2f}</b>{label}</div></div>")
    return "<div class='fb-wrap'>" + "".join(out) + "</div>"


def _shuffle() -> None:
    """Clear the search text and draw a fresh set of popular songs."""
    st.session_state["q_in"] = ""
    st.session_state["seed"] = st.session_state.get("seed", 1) + 1


def _search_box(cat: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([4, 1.3, 3])
    query = c1.text_input("Search", placeholder="Search a song or artist…", label_visibility="collapsed", key="q_in")
    c2.button("🎲 Shuffle", on_click=_shuffle, use_container_width=True)
    fair = c3.toggle("Only songs the model has never seen", value=True,
                     help="Those songs are in the held-out test set, so the result is a fair test. "
                          "Turn this off to search every song (songs used in training give optimistic results).")
    # Unseen songs make the prediction a fair test, so they are the default pool.
    pool = cat[cat["heldout"]] if fair else cat
    if query.strip():
        shown = _search(pool, query)
        st.caption(f"{len(shown)} match(es), most popular first." if len(shown) else "No match. Try another spelling, or turn the toggle off.")
    else:
        popular = pool[pool["popularity"] >= 55]
        shown = (popular if len(popular) >= 8 else pool).sample(min(8, len(pool)), random_state=st.session_state.get("seed", 1))
        st.caption("Popular songs to start with. Press Shuffle for more, or search above.")
    _bubble_buttons(shown, "q" if query.strip() else "s")


def _feature_chart(result: dict, feats: pd.Series, pred: str, truth: list[str]) -> str:
    gm = _genre_means(result)
    colors = {"This track": "#1DB954", f"Model's guess: {pred}": "#d0d0d0"}
    first = truth[0]
    if first != pred:
        colors[f"Dataset label: {first}"] = "#7a7a7a"
    rows = []
    for col, label in FEATURES:
        if col not in feats:
            continue
        v = {"This track": float(feats[col]), f"Model's guess: {pred}": float(gm.loc[pred, col])}
        if first != pred:
            v[f"Dataset label: {first}"] = float(gm.loc[first, col])
        rows.append({"l": label, "v": v})
    return legend(colors) + gbars(rows, colors, 1.0, lambda v, s: f"{v:.2f}")


def _reliability_chart(result: dict, pred: str, truth: list[str]) -> str:
    reports = result["test_reports_by_model"]
    colors = {m: c for m, c in MODEL_COLORS.items() if m in reports}
    genres = [pred] + [g for g in truth[:1] if g != pred]
    rows = [{"l": f"{g} ({'guess' if g == pred else 'dataset label'})",
             "v": {m: reports[m].get(g, {}).get("f1-score") for m in colors}} for g in genres]
    return legend(colors) + gbars(rows, colors, 1.0, lambda v, s: f"F1 {v:.2f}")


def render_picker(result: dict, raw: pd.DataFrame) -> None:
    cat = _catalog(result, raw)
    best = result["best_name"]

    st.caption("How to use: search a song, press Listen, then read the model's guess and the diagrams that explain it.")
    st.markdown("<div class='step'><b>1</b>Find a song</div>", unsafe_allow_html=True)
    _search_box(cat)

    tid = st.session_state.get("picked")
    if not tid or not cat["track_id"].eq(tid).any():
        st.info("Pick a song above to begin.")
        return

    track = cat[cat["track_id"] == tid].iloc[0]
    X_row = result["X"].loc[[track["row"]]]
    feats = X_row.iloc[0]

    st.markdown("<div class='step'><b>2</b>Listen</div>", unsafe_allow_html=True)
    badge = ("<span class='badge test'>Unseen by the models: fair test</span>" if track["heldout"] else
             "<span class='badge seen'>Seen in training: not a fair test</span>")
    st.markdown(f"<div class='np'><div class='disc'></div><div><h3>{html.escape(track['track_name'])}</h3>"
                f"<p>{html.escape(track['artists'].replace(';', ', '))} · {html.escape(str(track['album_name']))}</p>{badge}</div></div>",
                unsafe_allow_html=True)
    _player_block(tid)
    mins = feats["duration_ms"] / 60000 if "duration_ms" in feats else 0
    key = KEY_NAMES[int(feats["key"])] if "key" in feats and 0 <= int(feats["key"]) < 12 else "?"
    st.markdown(f"<span class='chip'>{feats.get('tempo', 0):.0f} BPM</span><span class='chip'>Key {key} "
                f"{'major' if feats.get('mode') == 1 else 'minor'}</span><span class='chip'>{mins:.1f} min</span>"
                f"<span class='chip'>Popularity {int(feats.get('popularity', 0))}</span>", unsafe_allow_html=True)
    st.markdown("**What the model hears.** Bigger bubble = higher value (0 to 1).")
    st.markdown(_feature_bubbles(feats), unsafe_allow_html=True)

    st.markdown("<div class='step'><b>3</b>The model's guess</div>", unsafe_allow_html=True)
    # One track ID can carry several genre labels in the source data, so a guess counts as correct if it matches any of them.
    truth = sorted(set(result["y"][st.session_state["_ids"].values == tid]))
    # The saved pipeline takes the same 15 features it was trained on, so no extra preprocessing is needed here.
    pred = result["best_pipeline"].predict(X_row)[0]
    ok = pred in truth
    others = "".join(f"<div class='mb'><small>{html.escape(n)} · held-out test</small><h4>{m['accuracy']:.0%} accuracy</h4>"
                     f"<small>macro F1 {m['macro_f1']:.3f}</small></div>"
                     for n, m in result["test_metrics_by_model"].items() if n != best)
    st.markdown(f"<div class='mb-row'><div class='mb sel'><small>{html.escape(best)} · selected model</small><h4>{html.escape(pred)}</h4>"
                f"{'<span class=hit>✓ Matches the dataset label</span>' if ok else '<span class=miss>✗ Different from the dataset label</span>'}</div>{others}</div>",
                unsafe_allow_html=True)
    st.caption("Only the selected model is saved, so live predictions come from it. The other two are shown by their held-out test scores.")
    st.markdown(f"<div class='panel'><b>Dataset label:</b> {html.escape(', '.join(truth))}"
                + (" <small>(this track ID has more than one label in the source data)</small>" if len(truth) > 1 else "") + "</div>",
                unsafe_allow_html=True)

    pipe = result["best_pipeline"]
    if hasattr(pipe, "predict_proba"):
        probs = pd.Series(pipe.predict_proba(X_row)[0], index=pipe.named_steps["classifier"].classes_).nlargest(5)
        st.markdown(f"**{best}: top five probability estimates**")
        st.markdown("".join(
            f"<div class='pbar'><span class='n'>{html.escape(g)}</span><div class='track'><div class='fill' style='width:{max(p * 100, 2):.0f}%'></div></div>"
            f"<span class='v'>{p:.0%}</span></div>" for g, p in probs.items()), unsafe_allow_html=True)
        st.caption("Model probability estimates, not guarantees. Accuracy is modest across 114 labels (see Compare).")

    st.markdown("<div class='step'><b>4</b>Why this guess?</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h3>This track against the average track of each genre</h3>"
                "<div class='cap'>The closer the green bar is to the light-grey one, the more the song resembles the genre the model picked. "
                "A bigger gap to the dark-grey one shows why the model did not choose the dataset label." +
                (" (First label shown; this track ID has several.)" if len(truth) > 1 else "") + "</div>"
                + _feature_chart(result, feats, pred, truth) + "</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h3>How reliable is each model on these genres?</h3>"
                "<div class='cap'>F1 on the held-out test set for the genre the model guessed (and the dataset label). "
                "Low bars mean the model rarely gets that genre right, so treat its guess with caution.</div>"
                + _reliability_chart(result, pred, truth) + "</div>", unsafe_allow_html=True)

    # Only unseen songs count toward the score, because the model already learned from the others.
    tally = st.session_state.setdefault("tally", {})
    tally[tid] = {"heldout": bool(track["heldout"]), "hit": ok}
    fair = [v for v in tally.values() if v["heldout"]]
    expected = result["test_metrics_by_model"][best]["accuracy"]
    st.markdown("<div class='step'><b>5</b>Your scoreboard</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score'><div class='sb'><b>{sum(v['hit'] for v in fair)}/{len(fair)}</b>your unseen picks matched</div>"
                f"<div class='sb'><b>{expected:.0%}</b>expected on the test set</div></div>", unsafe_allow_html=True)
    st.caption(f"Only unseen (held-out) tracks count: {len(fair)} of your {len(tally)} picks. "
               "A pick matches if it equals any label of that track. A handful of songs is an anecdote; the Compare tab has the real results.")
    if st.button("Reset scoreboard"):
        st.session_state["tally"] = {}
        st.rerun()
