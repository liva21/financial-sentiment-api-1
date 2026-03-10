import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from collections import Counter

API_URL = "http://localhost:8000"

COLORS = {
    "positive": "#00d4aa",
    "neutral":  "#6b7280",
    "negative": "#ff4757",
}

EMOJI = {
    "positive": "▲",
    "neutral":  "■",
    "negative": "▼",
}

RISK_COLOR = {
    "HIGH":   "#ff4757",
    "MEDIUM": "#ffa502",
    "LOW":    "#00d4aa",
}

EXAMPLE_SENTENCES = [
    "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the year-ago period.",
    "Stock prices crashed after company reported massive losses and declared bankruptcy.",
    "Net sales remained stable compared to the previous fiscal year.",
    "Revenue surged 40% driven by record-breaking demand across all segments.",
    "The firm announced layoffs affecting 2,000 employees amid declining revenues.",
    "The board decided to maintain the current dividend policy unchanged.",
    "BIST 100 rekor kirdi, yatirimcilar buyuk kazanc elde etti.",
    "Sirket batti, hissedarlar tum parasini kaybetti, borclar odenemiyor.",
    "Merkez Bankasi faiz oranini degistirmedi.",
]

st.set_page_config(
    page_title="FinSentiment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bloomberg-style dark theme CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

.stApp {
    background: #0a0e1a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2a3a;
}

[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #0a1628 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 20px 0 16px 0;
    margin-bottom: 24px;
}

.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #00d4aa;
    letter-spacing: -0.5px;
    margin: 0;
}

.main-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #4a6fa5;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #1e2a3a;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #4a6fa5 !important;
    background: transparent;
    border: none;
    padding: 12px 20px;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
    background: transparent !important;
}

/* Text area */
.stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 4px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}

.stTextArea textarea:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 1px #00d4aa20 !important;
}

/* Buttons */
.stButton button {
    background: #00d4aa !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    transition: all 0.15s ease !important;
}

.stButton button:hover {
    background: #00b894 !important;
    transform: translateY(-1px) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e2a3a;
    border-radius: 4px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4a6fa5 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #00d4aa !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #0d1117;
    border: 1px solid #1e2a3a;
    border-radius: 4px;
}

/* Selectbox */
.stSelectbox select, [data-baseweb="select"] {
    background: #0d1117 !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Divider */
hr {
    border-color: #1e2a3a !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #00d4aa !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4aa; }

/* Success/Error/Info */
.stSuccess { background: #00d4aa15 !important; border-color: #00d4aa !important; }
.stError { background: #ff475715 !important; border-color: #ff4757 !important; }
.stInfo { background: #4a6fa515 !important; border-color: #4a6fa5 !important; }
.stWarning { background: #ffa50215 !important; border-color: #ffa502 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">⬛ FINSENTIMENT</div>
    <div class="main-subtitle">Financial NLP Intelligence Platform · TR/EN</div>
</div>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
def call_predict(text):
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("API offline — uvicorn src.api:app --port 8000")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def call_batch(texts):
    try:
        r = requests.post(f"{API_URL}/predict/batch", json={"texts": texts}, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def sentiment_badge(sentiment, confidence):
    color = COLORS[sentiment]
    arrow = EMOJI[sentiment]
    return f"""<span style="
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px; font-weight: 600;
        color: {color};
        background: {color}15;
        border: 1px solid {color}40;
        padding: 4px 10px; border-radius: 3px;
        letter-spacing: 0.5px;">
        {arrow} {sentiment.upper()} {confidence:.1%}
    </span>"""

def risk_badge(risk_level):
    if not risk_level:
        return ""
    color = RISK_COLOR.get(risk_level, "#6b7280")
    return f"""<span style="
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; font-weight: 500;
        color: {color};
        background: {color}15;
        border: 1px solid {color}40;
        padding: 3px 8px; border-radius: 3px;
        letter-spacing: 0.5px;">
        RISK:{risk_level}
    </span>"""

def result_card(result):
    sentiment  = result["sentiment"]
    confidence = result["confidence"]
    language   = result.get("language", "en")
    risk_level = result.get("risk_level", "")
    risk_score = result.get("risk_score", 0)
    keywords   = result.get("keywords", [])
    translated = result.get("translated_text")
    latency    = result.get("latency_ms", 0)
    color      = COLORS[sentiment]
    neg        = result["scores"]["negative"]
    neu        = result["scores"]["neutral"]
    pos        = result["scores"]["positive"]

    # Translated row
    tr_row = ""
    if translated:
        tr_row = (
            "<div style='margin-top:12px;padding:10px 12px;"
            "background:#0a0e1a;border-left:2px solid #1e3a5f;"
            "font-family:monospace;font-size:11px;color:#4a6fa5;'>"
            "<span style='color:#1e3a5f;text-transform:uppercase;"
            "letter-spacing:1px;'>TR→EN</span>"
            f" &nbsp; {translated}</div>"
        )

    # Keyword row
    kw_row = ""
    if keywords:
        kw_spans = ""
        for kw in keywords:
            kw_spans += (
                f"<span style='font-family:monospace;font-size:10px;"
                f"color:#4a6fa5;background:#0a0e1a;border:1px solid #1e2a3a;"
                f"padding:2px 8px;border-radius:2px;margin-right:4px;'>{kw}</span>"
            )
        kw_row = f"<div style='margin-top:10px;'>{kw_spans}</div>"

    # Badge row
    s_badge  = sentiment_badge(sentiment, confidence)
    r_badge  = risk_badge(risk_level)

    html = (
        f"<div style='background:#0d1117;border:1px solid {color}30;"
        f"border-left:3px solid {color};border-radius:4px;"
        f"padding:20px 24px;margin:12px 0;'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;flex-wrap:wrap;gap:8px;'>"
        f"<div style='display:flex;align-items:center;gap:12px;'>"
        f"{s_badge}{r_badge}"
        f"<span style='font-family:monospace;font-size:10px;"
        f"color:#1e3a5f;letter-spacing:1px;'>"
        f"LANG:{language.upper()} &nbsp; {latency:.0f}ms</span></div>"
        f"<div style='display:flex;gap:16px;'>"
        f"<span style='font-family:monospace;font-size:11px;color:#ff4757;'>NEG {neg:.3f}</span>"
        f"<span style='font-family:monospace;font-size:11px;color:#6b7280;'>NEU {neu:.3f}</span>"
        f"<span style='font-family:monospace;font-size:11px;color:#00d4aa;'>POS {pos:.3f}</span>"
        f"</div></div>"
        f"{tr_row}{kw_row}</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

def mini_chart(scores, sentiment):
    color = COLORS[sentiment]
    labels = ["negative", "neutral", "positive"]
    values = [scores["negative"], scores["neutral"], scores["positive"]]
    colors = [COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#94a3b8"),
    ))
    fig.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        height=200,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(
            range=[0, 1.2],
            tickformat=".0%",
            gridcolor="#1e2a3a",
            tickfont=dict(family="IBM Plex Mono", size=10, color="#4a6fa5"),
        ),
        xaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11, color="#94a3b8")),
        showlegend=False,
    )
    return fig

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ANALYZE",
    "BATCH",
    "EXAMPLES",
    "MONITOR",
    "LIVE FEED",
])

# ── TAB 1: ANALYZE ────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                    letter-spacing:2px;color:#4a6fa5;text-transform:uppercase;
                    margin-bottom:8px;">
            INPUT TEXT
        </div>
        """, unsafe_allow_html=True)

        text_input = st.text_area(
            label="text",
            placeholder="Enter financial news or tweet...\nTürkçe metin de girebilirsiniz.",
            height=160,
            label_visibility="collapsed",
        )
        analyze_btn = st.button("▶ ANALYZE", use_container_width=True)

    with col2:
        if analyze_btn and text_input.strip():
            with st.spinner("Processing..."):
                result = call_predict(text_input)
            if result:
                result_card(result)
                st.plotly_chart(mini_chart(result["scores"], result["sentiment"]),
                               use_container_width=True)
        elif analyze_btn:
            st.warning("Input required.")

        if not analyze_btn:
            st.markdown("""
            <div style="border:1px dashed #1e2a3a; border-radius:4px; padding:40px 20px;
                        text-align:center; margin-top:8px;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                            letter-spacing:2px;color:#1e3a5f;text-transform:uppercase;">
                    Awaiting input
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: BATCH ──────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                letter-spacing:2px;color:#4a6fa5;text-transform:uppercase;
                margin-bottom:8px;">
        BATCH INPUT — ONE TEXT PER LINE (MAX 32)
    </div>
    """, unsafe_allow_html=True)

    batch_input = st.text_area(
        label="batch",
        placeholder="Line 1...\nLine 2...\nLine 3...",
        height=160,
        label_visibility="collapsed",
    )
    batch_btn = st.button("▶ RUN BATCH", use_container_width=True)

    if batch_btn and batch_input.strip():
        texts = [t.strip() for t in batch_input.strip().split("\n") if t.strip()]
        if len(texts) > 32:
            st.error("MAX 32 LINES")
        else:
            with st.spinner(f"Processing {len(texts)} texts..."):
                result = call_batch(texts)
            if result:
                rows = []
                for r in result["results"]:
                    rows.append({
                        "TEXT"      : r["text"][:70] + ("..." if len(r["text"]) > 70 else ""),
                        "LANG"      : r.get("language", "en").upper(),
                        "SENTIMENT" : f"{EMOJI[r['sentiment']]} {r['sentiment'].upper()}",
                        "CONF"      : f"{r['confidence']:.1%}",
                        "RISK"      : r.get("risk_level", ""),
                        "KEYWORDS"  : ", ".join(r.get("keywords", [])[:3]),
                    })

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()
                sentiments = [r["sentiment"] for r in result["results"]]
                counts = Counter(sentiments)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("TOTAL", len(texts))
                c2.metric("▲ POSITIVE", counts.get("positive", 0))
                c3.metric("■ NEUTRAL",  counts.get("neutral",  0))
                c4.metric("▼ NEGATIVE", counts.get("negative", 0))

# ── TAB 3: EXAMPLES ───────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                letter-spacing:2px;color:#4a6fa5;text-transform:uppercase;
                margin-bottom:16px;">
        SAMPLE FINANCIAL TEXTS
    </div>
    """, unsafe_allow_html=True)

    for i, sentence in enumerate(EXAMPLE_SENTENCES):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                        color:#94a3b8;padding:12px 16px;
                        background:#0d1117;border:1px solid #1e2a3a;
                        border-radius:3px;margin-bottom:4px;">
                {sentence}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("▶ RUN", key=f"ex_{i}"):
                with st.spinner(""):
                    result = call_predict(sentence)
                if result:
                    s = result["sentiment"]
                    c = result["confidence"]
                    color = COLORS[s]
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;'
                        f'font-size:12px;color:{color};font-weight:600;">'
                        f'{EMOJI[s]} {s.upper()}<br>'
                        f'<span style="font-size:10px;color:#4a6fa5;">{c:.1%}</span></div>',
                        unsafe_allow_html=True
                    )

# ── TAB 4: MONITOR ────────────────────────────────────────────
with tab4:
    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("↻ REFRESH", use_container_width=True):
            st.rerun()

    try:
        stats = requests.get(f"{API_URL}/monitoring/stats", timeout=5).json()
    except:
        st.error("API offline")
        stats = {}

    if not stats or stats.get("total", 0) == 0:
        st.markdown("""
        <div style="text-align:center;padding:60px;
                    font-family:'IBM Plex Mono',monospace;
                    font-size:11px;letter-spacing:2px;
                    color:#1e3a5f;text-transform:uppercase;">
            No requests recorded yet
        </div>
        """, unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TOTAL REQUESTS", stats["total"])
        if stats.get("latency"):
            c2.metric("AVG LATENCY", f"{stats['latency'][0]['avg_ms']} ms")
        if stats.get("distribution"):
            dist = {d["sentiment"]: d["count"] for d in stats["distribution"]}
            c3.metric("▲ POSITIVE", dist.get("positive", 0))
            c4.metric("▼ NEGATIVE", dist.get("negative", 0))

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            if stats.get("distribution"):
                df_dist = pd.DataFrame(stats["distribution"])
                fig = go.Figure(go.Pie(
                    labels=df_dist["sentiment"],
                    values=df_dist["count"],
                    marker=dict(
                        colors=[COLORS.get(s, "#6b7280") for s in df_dist["sentiment"]],
                        line=dict(color="#0a0e1a", width=2),
                    ),
                    hole=0.6,
                    textfont=dict(family="IBM Plex Mono", size=11),
                ))
                fig.update_layout(
                    title=dict(text="SENTIMENT DISTRIBUTION",
                              font=dict(family="IBM Plex Mono", size=10,
                                       color="#4a6fa5"), x=0),
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    height=280, margin=dict(t=40, b=10),
                    legend=dict(font=dict(family="IBM Plex Mono", size=10,
                                         color="#94a3b8")),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if stats.get("hourly"):
                df_h = pd.DataFrame(stats["hourly"])
                fig2 = go.Figure(go.Bar(
                    x=df_h["hour"], y=df_h["count"],
                    marker_color="#00d4aa",
                    marker_line_width=0,
                ))
                fig2.update_layout(
                    title=dict(text="REQUEST VOLUME (24H)",
                              font=dict(family="IBM Plex Mono", size=10,
                                       color="#4a6fa5"), x=0),
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    height=280, margin=dict(t=40, b=10),
                    xaxis=dict(tickangle=45, gridcolor="#1e2a3a",
                              tickfont=dict(family="IBM Plex Mono", size=9,
                                           color="#4a6fa5")),
                    yaxis=dict(gridcolor="#1e2a3a",
                              tickfont=dict(family="IBM Plex Mono", size=9,
                                           color="#4a6fa5")),
                )
                st.plotly_chart(fig2, use_container_width=True)

        if stats.get("recent"):
            st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                        letter-spacing:2px;color:#4a6fa5;text-transform:uppercase;
                        margin:16px 0 8px 0;">
                RECENT REQUESTS
            </div>
            """, unsafe_allow_html=True)
            df_r = pd.DataFrame(stats["recent"])
            df_r["text"]       = df_r["text"].str[:60] + "..."
            df_r["confidence"] = df_r["confidence"].apply(lambda x: f"{x:.1%}")
            df_r["latency_ms"] = df_r["latency_ms"].apply(lambda x: f"{x:.0f}ms")
            st.dataframe(df_r, use_container_width=True, hide_index=True)

# ── TAB 5: LIVE FEED ──────────────────────────────────────────
with tab5:
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("↻ FETCH NEWS", use_container_width=True):
            with st.spinner("Fetching..."):
                try:
                    from src.news_collector import fetch_all_feeds, init_news_db
                    init_news_db()
                    fetch_all_feeds()
                    st.success("Updated")
                except Exception as e:
                    st.error(f"{e}")
            st.rerun()
    with col2:
        risk_filter = st.selectbox(
            "FILTER",
            ["ALL", "HIGH", "MEDIUM", "LOW"],
            label_visibility="collapsed",
        )

    try:
        from src.news_collector import get_recent_news, init_news_db
        init_news_db()
        news = get_recent_news(limit=50)
    except Exception as e:
        st.error(f"{e}")
        news = []

    if not news:
        st.markdown("""
        <div style="text-align:center;padding:60px;
                    font-family:'IBM Plex Mono',monospace;
                    font-size:11px;letter-spacing:2px;
                    color:#1e3a5f;text-transform:uppercase;">
            No news fetched yet — click FETCH NEWS
        </div>
        """, unsafe_allow_html=True)
    else:
        if risk_filter != "ALL":
            news = [n for n in news if n.get("risk_level") == risk_filter]

        sentiments = [n["sentiment"] for n in news if n.get("sentiment")]
        counts = Counter(sentiments)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TOTAL",      len(news))
        c2.metric("▲ POSITIVE", counts.get("positive", 0))
        c3.metric("■ NEUTRAL",  counts.get("neutral",  0))
        c4.metric("▼ NEGATIVE", counts.get("negative", 0))

        st.divider()

        for item in news:
            sentiment  = item.get("sentiment", "neutral")
            risk_level = item.get("risk_level", "")
            keywords   = item.get("keywords", [])
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except:
                    keywords = []

            color    = COLORS.get(sentiment, "#6b7280")
            risk_col = RISK_COLOR.get(risk_level, "#6b7280")
            title    = item.get("title", "")[:120]
            source   = item.get("source", "")
            pub      = str(item.get("published", ""))[:16]
            url      = item.get("url", "#")
            conf     = item.get("confidence", 0)
            emj      = EMOJI.get(sentiment, "")

            # Keyword spans
            kw_spans = ""
            for kw in keywords[:4]:
                kw_spans += (
                    f"<span style='font-size:9px;font-family:monospace;"
                    f"color:#4a6fa5;background:#0a0e1a;border:1px solid #1e2a3a;"
                    f"padding:1px 6px;border-radius:2px;margin-right:3px;'>{kw}</span>"
                )
            kw_row = f"<div style='margin-top:6px;'>{kw_spans}</div>" if kw_spans else ""

            # Risk span
            risk_span = ""
            if risk_level:
                risk_span = (
                    f"<span style='font-size:9px;font-family:monospace;"
                    f"color:{risk_col};background:{risk_col}15;"
                    f"border:1px solid {risk_col}40;"
                    f"padding:1px 6px;border-radius:2px;margin-left:8px;'>"
                    f"RISK:{risk_level}</span>"
                )

            html = (
                f"<div style='border-left:2px solid {color};padding:10px 16px;"
                f"margin-bottom:6px;background:#0d1117;border-radius:0 3px 3px 0;'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;flex-wrap:wrap;gap:4px;'>"
                f"<span style='font-family:monospace;font-size:9px;"
                f"letter-spacing:1px;color:#1e3a5f;text-transform:uppercase;'>"
                f"{source} · {pub}</span>"
                f"<span>"
                f"<span style='font-family:monospace;font-size:10px;"
                f"font-weight:600;color:{color};'>"
                f"{emj} {sentiment.upper()} {conf:.0%}</span>"
                f"{risk_span}</span></div>"
                f"<div style='font-size:13px;font-weight:500;"
                f"color:#e2e8f0;margin-top:6px;line-height:1.4;'>"
                f"<a href='{url}' target='_blank' "
                f"style='text-decoration:none;color:#e2e8f0;'>{title}</a></div>"
                f"{kw_row}</div>"
            )
            st.markdown(html, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                letter-spacing:2px;color:#00d4aa;text-transform:uppercase;
                font-weight:600;margin-bottom:16px;">
        ⬛ FINSENTIMENT
    </div>
    """, unsafe_allow_html=True)

    try:
        r    = requests.get(f"{API_URL}/health", timeout=3)
        data = r.json()
        st.markdown(
            "<div style='font-family:monospace;font-size:10px;"
            "color:#00d4aa;letter-spacing:1px;margin-bottom:12px;'>"
            "● SYSTEM ONLINE</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='font-family:monospace;font-size:9px;"
            f"color:#4a6fa5;line-height:2;'>"
            f"STATUS &nbsp;&nbsp; ok<br>"
            f"MODEL &nbsp;&nbsp;&nbsp; finbert-finetuned<br>"
            f"DEVICE &nbsp;&nbsp; {data.get('device', 'cpu').upper()}<br>"
            f"</div>",
            unsafe_allow_html=True
        )
    except:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                    color:#ff4757;letter-spacing:1px;">
            ● SYSTEM OFFLINE
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
                color:#1e3a5f;letter-spacing:1px;line-height:2;">
        MODEL &nbsp;&nbsp;&nbsp; ProsusAI/FinBERT<br>
        DATASET &nbsp; financial_phrasebank<br>
        F1 SCORE &nbsp;0.963<br>
        ACCURACY 0.978<br>
        LATENCY &nbsp; ~300ms (EN)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ~1500ms (TR)<br>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
                color:#1e3a5f;letter-spacing:1px;line-height:2;">
        LANG &nbsp;&nbsp; TR → Helsinki + FinBERT<br>
        LANG &nbsp;&nbsp; EN → FinBERT direct<br>
    </div>
    """, unsafe_allow_html=True)
