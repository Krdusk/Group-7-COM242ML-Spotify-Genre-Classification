"""Spotify-style theme: one green accent on black and greys, large readable text."""

THEME_CSS = """
<style>
:root { color-scheme: dark; --g:#1DB954; --bg:#121212; --card:#181818; --card2:#242424; --line:#2f2f2f; --mut:#b3b3b3; --tx:#f2f2f2; }
.stApp, [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--tx); }
@keyframes pop { from { transform:scale(.7); opacity:0 } to { transform:scale(1); opacity:1 } }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stStatusWidget"] { display:none !important; }
.block-container { position:relative; z-index:1; max-width:1080px; padding:1.2rem 1.6rem 4rem; }
/* text */
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th { font-size:1.02rem; line-height:1.6; color:var(--tx); }
.stApp h1,.stApp h2,.stApp h3,.stApp h4 { color:#fff; letter-spacing:-.01em; line-height:1.25; }
.stApp h2 { font-size:1.6rem; margin-top:1.4rem; } .stApp h3 { font-size:1.2rem; }
[data-testid="stCaptionContainer"] p, small, .cap { color:var(--mut) !important; font-size:.92rem !important; }
a { color:#fff; text-decoration-color:var(--g); }
/* hero */
.hero { display:flex; gap:1.1rem; align-items:center; padding:1.2rem 1.6rem; border-radius:28px; margin:0 0 .8rem;
  background:var(--card); border:1px solid var(--line); }
.hero .logo { flex:none; width:60px; height:60px; border-radius:50%; background:var(--g); color:#000; display:grid; place-items:center; font-size:1.8rem; box-shadow:0 0 0 8px rgba(29,185,84,.15); margin:0 .5rem 0 .2rem; }
.hero h1 { font-size:clamp(1.5rem,3vw,2rem); margin:0 0 .2rem; } .hero p { margin:0 0 .5rem; color:#d9d9d9; }
.chip { display:inline-block; padding:.2rem .8rem; margin:0 .35rem .3rem 0; border-radius:999px; background:rgba(255,255,255,.09); font-size:.85rem; color:#eee; }
/* navigation: sticky pill tabs */
[role="tablist"] { position:sticky; top:0; z-index:50; gap:.4rem; flex-wrap:wrap; border:0; padding:.55rem .2rem; background:rgba(18,18,18,.92); backdrop-filter:blur(8px); border-radius:0 0 18px 18px; }
[data-testid="stTab"], .stTabs button[role="tab"] { background:var(--card2) !important; color:#fff !important; border-radius:999px !important; padding:.45rem 1.1rem !important; height:auto !important; border:1px solid #3a3a3a !important; }
[data-testid="stTab"] p { color:inherit !important; font-size:.95rem !important; font-weight:600; }
[data-testid="stTab"][aria-selected="true"], .stTabs button[role="tab"][aria-selected="true"] { background:var(--g) !important; color:#000 !important; border-color:var(--g) !important; }
[role="tablist"] > div[aria-hidden], [data-testid="stTabHighlight"], [data-testid="stTabBorder"], .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
/* inputs and buttons */
[data-testid="stTextInputRootElement"], [data-baseweb="input"], [data-baseweb="select"] > div { background:#2a2a2a !important; border-radius:999px !important; border:1px solid #4a4a4a !important; overflow:hidden; }
[data-testid="stTextInputRootElement"]:focus-within { border-color:var(--g) !important; }
[data-testid="stTextInputRootElement"] input, [data-baseweb="input"] input { padding-left:1.1rem !important; color:#fff !important; font-size:1rem; }
[data-baseweb="popover"],[role="listbox"],[role="option"] { background:#242424 !important; color:#fff !important; }
div.stButton > button, div[data-testid="stFormSubmitButton"] button { color:#fff; background:var(--card2); border:1px solid #3a3a3a; border-radius:999px; padding:.5rem 1.2rem; transition:.15s; font-size:1rem; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover { background:var(--g); color:#000; border-color:var(--g); }
div[data-testid="stFormSubmitButton"] button { background:var(--g); color:#000; font-weight:700; }
/* alerts: neutral card with a green edge */
[data-testid="stAlert"] { background:#1b1b1b !important; border:1px solid var(--line) !important; border-left:4px solid var(--g) !important; border-radius:16px !important; }
[data-testid="stAlert"] > div, [data-testid="stAlert"] div { background:transparent !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] svg { color:var(--tx) !important; fill:var(--tx) !important; }
[data-testid="stDataFrame"], [data-testid="stExpander"] { border:1px solid var(--line); border-radius:18px; }
div[data-testid="stMetric"], .section-card, .model-card, .panel { padding:1rem 1.2rem; border-radius:22px; background:var(--card); border:1px solid var(--line); margin:.4rem 0 .8rem; }
div[data-testid="stMetricLabel"] p { color:var(--mut) !important; }
hr { border-color:var(--line); }
/* picker */
.step { display:flex; align-items:center; gap:.7rem; margin:1.6rem 0 .5rem; font-weight:700; font-size:1.25rem; }
.step b { width:32px; height:32px; border-radius:50%; background:var(--g); color:#000; display:grid; place-items:center; font-size:1rem; flex:none; }
.np { display:flex; gap:1rem; align-items:center; padding:.4rem 0; animation:pop .3s; }
.np .disc { flex:none; width:60px; height:60px; border-radius:50%; background:radial-gradient(circle,#121212 0 14%,var(--g) 15% 100%); }
.np h3 { margin:0; font-size:1.35rem; } .np p { margin:.1rem 0 0; color:var(--mut); }
.badge { display:inline-block; padding:.2rem .8rem; border-radius:999px; font-size:.82rem; font-weight:700; margin-top:.5rem; }
.badge.test { background:var(--g); color:#000; } .badge.seen { background:transparent; color:#ddd; border:1px solid #666; }
.fb-wrap { display:flex; flex-wrap:wrap; gap:.7rem; align-items:center; justify-content:center; padding:.6rem 0; min-height:150px; }
.fb { border-radius:50%; display:grid; place-items:center; text-align:center; color:#fff; font-size:.75rem; line-height:1.15; animation:pop .45s backwards; border:1px solid rgba(29,185,84,.6); }
.fb b { display:block; font-size:1rem; }
.mb-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.8rem; }
.mb { padding:1rem 1.1rem; border-radius:24px; background:var(--card); border:1px solid var(--line); }
.mb.sel { border:2px solid var(--g); } .mb small { display:block; color:var(--mut); font-size:.88rem; } .mb h4 { margin:.25rem 0; font-size:1.3rem; word-break:break-word; }
.hit { color:var(--g); font-weight:700; } .miss { color:#fff; font-weight:700; }
.pbar { display:flex; align-items:center; gap:.7rem; margin:.4rem 0; }
.pbar span.n { width:140px; flex:none; font-size:.95rem; text-align:right; color:var(--tx) !important; }
.pbar .track { flex:1; height:20px; border-radius:999px; background:#2a2a2a; overflow:hidden; } .pbar .fill { height:100%; border-radius:999px; background:var(--g); }
.pbar span.v { width:48px; font-size:.9rem; color:var(--mut); }
.score { display:flex; gap:.8rem; flex-wrap:wrap; } .score .sb { flex:1; min-width:150px; text-align:center; padding:.8rem; border-radius:24px; background:var(--card); border:1px solid var(--line); }
.score .sb b { font-size:1.6rem; color:var(--g); display:block; }
/* charts */
.lg { display:flex; gap:1.1rem; flex-wrap:wrap; font-size:.9rem; color:var(--mut); margin:.2rem 0 .6rem; }
.lg i { display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:.4rem; vertical-align:-1px; }
.card { padding:1.1rem 1.3rem; border-radius:24px; background:var(--card); border:1px solid var(--line); margin:.8rem 0; } .card h3 { margin:.1rem 0 .2rem; font-size:1.1rem; }
.gr { display:grid; grid-template-columns:150px 1fr; gap:.6rem; align-items:center; margin:.6rem 0; } .gl { font-size:.92rem; text-align:right; color:var(--tx); } .gb { display:grid; gap:3px; }
.bw { display:flex; align-items:center; gap:.5rem; } .tr { position:relative; flex:1; height:14px; border-radius:999px; background:#262626; } .tr .b { height:100%; border-radius:999px; }
.wk { position:absolute; top:50%; height:2px; background:#fff; transform:translateY(-50%); }
.wk::before,.wk::after { content:""; position:absolute; top:-4px; width:2px; height:10px; background:#fff; } .wk::before { left:0; } .wk::after { right:0; }
.bv { width:104px; flex:none; font-size:.85rem; color:var(--mut); }
.hm { display:flex; flex-wrap:wrap; gap:1rem; justify-content:center; } .hm figure { margin:0; flex:1 1 320px; max-width:440px; }
.hg { display:grid; gap:2px; } .hx { aspect-ratio:1; border-radius:4px; } .hc { writing-mode:vertical-rl; transform:rotate(180deg); font-size:.66rem; color:#d0d0d0; height:84px; text-align:left; }
.hr { font-size:.66rem; color:#d0d0d0; text-align:right; align-self:center; padding-right:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.hm figcaption { font-size:.88rem; color:var(--mut); text-align:center; margin-top:.4rem; }
.vd li { margin:.5rem 0; }
@media (max-width:720px) { .hero { padding:1rem; border-radius:22px; } .hero .logo { width:46px; height:46px; font-size:1.3rem; }
  .pbar span.n { width:96px; } .gr { grid-template-columns:1fr; } .gl { text-align:left; } .block-container { padding:1rem .9rem 3rem; } }
</style>
"""

HERO_HTML = """
<div class="hero"><div class="logo">♫</div><div>
<h1>Spotify Genre Lab</h1>
<p>Search a song, listen, and see how a machine-learning model guesses its genre. Start on <b>Pick a track</b>; the other tabs show how the models were compared.</p>
<span class="chip">3 models</span><span class="chip">114 genres</span><span class="chip">5-fold cross-validation</span><span class="chip">held-out test set</span>
</div></div>
"""
