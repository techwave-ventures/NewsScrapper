from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests
import httpx
import asyncio
import os
import re
import time

# === Load Environment Variables ===
load_dotenv()
MONGO_URL = os.getenv("MONGO_URL")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")
HAILING_ENDPOINT = os.getenv("HAILING_ENDPOINT", "/hailing")
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "300"))  # Default: 5 min

# === MongoDB Setup ===
client = AsyncIOMotorClient(MONGO_URL)
db = client["Techsocial"]
collection = db["news-articles"]

# === FastAPI App ===
app = FastAPI()

# === Pydantic Model ===
class Article(BaseModel):
    headline: str
    bannerImage: Optional[str] = None
    article: str
    ref: str
    category: List[str]
    timestamp: Optional[str] = None

# === Category Map ===
CATEGORY_MAP = {
    "economy": "economy", "policy": "policy", "finance": "finance", "banking": "finance",
    "markets": "markets", "stocks": "stocks", "ipos": "ipos", "bonds": "bonds",
    "crypto": "crypto", "forex": "forex", "industry": "industry", "auto": "auto",
    "retail": "retail", "real-estate": "real-estate", "energy": "energy", "telecom": "telecom",
    "manufacturing": "manufacturing", "technology": "technology", "tech": "technology",
    "ai": "technology", "startups": "startups", "science": "science", "defence": "defence",
    "politics": "politics", "elections": "politics", "world": "world-news", "international": "world-news",
    "india": "india", "wealth": "wealth", "education": "education", "career": "career", "jobs": "career",
    "opinion": "opinion", "sme": "sme", "nri": "nri", "panache": "lifestyle", "environment": "environment",
}

# === Helpers ===
def detect_categories(href: str, headline: str) -> List[str]:
    segments = urlparse(href).path.lower().split("/") + [headline.lower()]
    combined = " ".join(segments)
    detected = {val for key, val in CATEGORY_MAP.items() if key.replace("-", "") in combined.replace("-", "")}
    return list(detected or ["general"])

def extract_full_article_text(soup: BeautifulSoup) -> str:
    blocks = soup.find_all("div", class_="Normal") or soup.find_all("p")
    if not blocks:
        return "No article content found."
    stop_phrases = [
        "catch all the", "subscribe to", "download et app",
        "find this comment offensive", "choose your reason",
        "hot on web", "in case you missed it",
        "top story", "top slideshow", "next read",
        "more less", "continue reading", "you can now subscribe",
        "read more on", "popular categories", "hot on web",
        "today’s newsquick reads", "web stories",
        "powered by", "terms of use", "privacy policy",
        "copyright ©", "stock radar", "elevate your knowledge", "etprime"
    ]

    text_chunks = []
    for tag in blocks:
        txt = tag.get_text(strip=True, separator=" ")
        if any(p in txt.lower() for p in stop_phrases):
            break
        text_chunks.append(txt)
    return re.sub(r"\s+", " ", " ".join(text_chunks)).strip() or "No article content found."

# === Scraping Logic ===
async def fetch_and_process_articles(batch_size: int = 15):
    print("🔄 Scraping latest articles...")
    url = "https://economictimes.indiatimes.com/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    base_url = "https://economictimes.indiatimes.com"
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=1)

    soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
    seen, hrefs = set(), []

    for a in soup.find_all("a", href=True):
        href = a['href'].strip()
        if href.startswith("/"):
            href = base_url + href
        elif not href.startswith("http"):
            href = "https://" + href.lstrip("/")
        if "/articleshow/" not in href or href in seen:
            continue
        seen.add(href)
        hrefs.append(href)

    articles = []

    async def process_url(href: str, client: httpx.AsyncClient):
        try:
            res = await client.get(href, timeout=10)

            # ✅ Skip dead links
            if res.status_code == 404:
                print(f"⛔ 404 Not Found: {href}")
                return
            elif res.status_code != 200:
                print(f"⚠️ Skipping ({res.status_code}): {href}")
                return

            soup = BeautifulSoup(res.text, "html.parser")
            title_tag = soup.find("meta", property="og:title")
            image_tag = soup.find("meta", property="og:image")
            date_meta = (
                soup.find("meta", itemprop="datePublished") or
                soup.find("meta", property="article:published_time")
            )
            headline = title_tag["content"] if title_tag else "No Title"
            banner = image_tag["content"] if image_tag else None
            timestamp = date_meta["content"] if date_meta else None
            pub_time = datetime.strptime(timestamp[:19], "%Y-%m-%dT%H:%M:%S") if timestamp else now

            if pub_time < cutoff:
                return

            summary = extract_full_article_text(soup)
            categories = detect_categories(href, headline)

            doc = {
                "headline": headline.strip(),
                "bannerImage": banner,
                "article": summary[:1000],
                "ref": href,
                "category": categories,
                "timestamp": pub_time.strftime("%Y-%m-%d %H:%M:%S")
            }

            if not await collection.find_one({"ref": href}):
                await collection.insert_one(doc)
                print("✅ Inserted:", href)
            else:
                print("📦 Already exists:", href)

            articles.append(doc)

        except Exception as e:
            print("❌ Error:", href, str(e))

    async with httpx.AsyncClient() as client:
        for i in range(0, len(hrefs), batch_size):
            batch = hrefs[i:i + batch_size]
            await asyncio.gather(*[process_url(h, client) for h in batch])

    print(f"📤 Total Articles Collected: {len(articles)}")
    return articles

# === Public Trigger ===
@app.get("/latest-news", response_model=List[Article])
async def get_news(interests: Optional[List[str]] = Query(None)):
    return await fetch_and_process_articles()

# === Hailing Route ===
@app.get(HAILING_ENDPOINT)
async def hailing_route():
    now_str = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    print(f"👋 Hailing route hit at {now_str}")
    return {"status": "OK", "message": "Server is alive", "timestamp": now_str}

# === Heartbeat + Scraping Task ===
async def heartbeat_task():
    await asyncio.sleep(10)
    hailing_url = f"{APP_BASE_URL}{HAILING_ENDPOINT}"
    print(f"🔁 Heartbeat started. Pinging every {HEARTBEAT_INTERVAL_SECONDS // 60} min...")

    async with httpx.AsyncClient() as client:
        while True:
            try:
                res = await client.get(hailing_url, timeout=10.0)
                print(f"✅ Hail pinged: {res.status_code} — {res.json().get('message')}")
                await fetch_and_process_articles()
            except Exception as e:
                print(f"⚠️ Heartbeat Error: {e}")
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

# === On Startup ===
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(heartbeat_task())
    print("🚀 Background task started.")
