import feedparser
import sqlite3
import schedule
import time
import os
import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

RSS_FEEDS = {
    "cnbc_finance": {
        "url"     : "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
        "name"    : "CNBC Finance",
        "lang"    : "en",
        "fulltext": False,   # CNBC summary zaten yeterli
    },
    "hurriyet_ekonomi": {
        "url"     : "https://www.hurriyet.com.tr/rss/ekonomi",
        "name"    : "Hurriyet Ekonomi",
        "lang"    : "tr",
        "fulltext": True,    # Tam metin çek
    },
}

API_URL = os.getenv("API_URL", "http://localhost:8000")
DB_PATH = os.getenv("DB_PATH", "data/monitoring.db")

# ── DB ───────────────────────────────────────────────────────

def _conn():
    os.makedirs("data", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_news_db():
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            guid       TEXT    UNIQUE,
            source     TEXT    NOT NULL,
            title      TEXT    NOT NULL,
            summary    TEXT,
            url        TEXT,
            published  TEXT,
            fetched_at TEXT    NOT NULL,
            sentiment  TEXT,
            confidence REAL,
            risk_score REAL,
            risk_level TEXT,
            keywords   TEXT,
            translated TEXT
        )
    """)
    c.commit()
    c.close()

def is_already_fetched(guid):
    c   = _conn()
    row = c.execute("SELECT id FROM news WHERE guid = ?", (guid,)).fetchone()
    c.close()
    return row is not None

def save_news(item):
    c = _conn()
    c.execute("""
        INSERT OR IGNORE INTO news
            (guid, source, title, summary, url, published,
             fetched_at, sentiment, confidence, risk_score,
             risk_level, keywords, translated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        item["guid"],
        item["source"],
        item["title"],
        item.get("summary", "")[:500],
        item.get("url", ""),
        item.get("published", ""),
        datetime.now().isoformat(),
        item.get("sentiment"),
        item.get("confidence"),
        item.get("risk_score"),
        item.get("risk_level"),
        json.dumps(item.get("keywords", []), ensure_ascii=False),
        item.get("translated_text"),
    ))
    c.commit()
    c.close()

def get_recent_news(limit=50):
    c    = _conn()
    rows = c.execute("""
        SELECT source, title, summary, url, published, fetched_at,
               sentiment, confidence, risk_score, risk_level, keywords
        FROM news ORDER BY fetched_at DESC LIMIT ?
    """, (limit,)).fetchall()
    c.close()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["keywords"] = json.loads(d["keywords"] or "[]")
        except:
            d["keywords"] = []
        result.append(d)
    return result

# ── Full text çekme ──────────────────────────────────────────

def fetch_full_text(url):
    """Haber sayfasından tam metni çek."""
    try:
        r    = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")

        # Önce article tag'ini dene
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text = " ".join(
            p.get_text(strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 40   # kısa/boş paragrafları atla
        )
        return text[:1500] if text else None
    except Exception as e:
        print(f"  Tam metin hatasi: {e}")
        return None

# ── API ──────────────────────────────────────────────────────

def analyze_text(text):
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"text": text[:1000]},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  API hatasi: {e}")
        return None

# ── Collector ────────────────────────────────────────────────

def fetch_feed(source_key, feed_config):
    print(f"\n Cekiliyor: {feed_config['name']}")
    try:
        r    = requests.get(feed_config["url"], headers=HEADERS, timeout=15)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
    except Exception as e:
        print(f"  Feed hatasi: {e}")
        return 0

    new_count = 0
    for entry in feed.entries[:8]:
        guid = getattr(entry, "id", entry.get("link", entry.get("title", "")))
        if is_already_fetched(guid):
            continue

        title   = getattr(entry, "title", "")
        summary = getattr(entry, "summary", "")
        url     = getattr(entry, "link", "")

        # Tam metin çek (Türkçe haberler için)
        if feed_config.get("fulltext") and url:
            full_text = fetch_full_text(url)
            if full_text and len(full_text) > len(summary):
                analysis_text = f"{title}. {full_text}"
                print(f"  -> {title[:60]}... [{len(full_text)} karakter tam metin]")
            else:
                analysis_text = f"{title}. {summary}"
                print(f"  -> {title[:60]}... [summary]")
        else:
            analysis_text = f"{title}. {summary}"
            print(f"  -> {title[:60]}...")

        result = analyze_text(analysis_text[:1000])

        item = {
            "guid"     : guid,
            "source"   : feed_config["name"],
            "title"    : title,
            "summary"  : summary,
            "url"      : url,
            "published": getattr(entry, "published", ""),
        }

        if result:
            item.update({
                "sentiment"      : result["sentiment"],
                "confidence"     : result["confidence"],
                "risk_score"     : result.get("risk_score"),
                "risk_level"     : result.get("risk_level"),
                "keywords"       : result.get("keywords", []),
                "translated_text": result.get("translated_text"),
            })
            emoji = {"positive": "📈", "neutral": "➖", "negative": "📉"}
            print(f"     {emoji.get(result['sentiment'], '?')} "
                  f"{result['sentiment']} ({result['confidence']:.1%}) "
                  f"| Risk: {result.get('risk_level', 'N/A')}"
                  f"| Keywords: {result.get('keywords', [])[:3]}")

        save_news(item)
        new_count += 1

    print(f"  {new_count} yeni haber kaydedildi")
    return new_count

def fetch_all_feeds():
    init_news_db()
    print(f"\n{'='*55}")
    print(f"  HABER GUNCELLEME — {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*55}")
    total = 0
    for key, config in RSS_FEEDS.items():
        total += fetch_feed(key, config)
    print(f"\n Toplam {total} yeni haber islendi.")

if __name__ == "__main__":
    fetch_all_feeds()
    schedule.every(30).minutes.do(fetch_all_feeds)
    print("\n Scheduler basladi — her 30 dakikada guncelleniyor.")
    while True:
        schedule.run_pending()
        time.sleep(60)
