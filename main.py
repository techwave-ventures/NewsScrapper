from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from motor.motor_asyncio import AsyncIOMotorClient

from urllib.parse import urlparse
from dotenv import load_dotenv
import requests
import httpx
import asyncio
import os
import re
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time

# === Load Environment Variables ===
load_dotenv()
MONGO_URL = os.getenv("MONGO_URL")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")
HAILING_ENDPOINT = os.getenv("HAILING_ENDPOINT", "/hailing")
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "300"))  # Default: 5 min

# === MongoDB Setup ===
client = AsyncIOMotorClient(MONGO_URL)
db = client["connectx"]
collection = db["news-articles"]
# New collection for storing subscriber emails
subscribers_collection = db["subscribers"]

# === FastAPI App ===
app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Pydantic Model ===
class Article(BaseModel):
    headline: str
    bannerImage: Optional[str] = None
    article: str
    ref: str
    category: List[str]
    timestamp: Optional[str] = None


# New Pydantic model for a subscriber
class Subscriber(BaseModel):
    email: str

# === Category Maps ===
CATEGORY_MAP = {
    # Economy & Finance
    "economy": "economy",
    "policy": "policy",
    "finance": "finance",
    "banking": "banking",
    "markets": "markets",
    "stocks": "stocks",
    "ipos": "ipos",
    "bonds": "bonds",
    "forex": "forex",
    "mutual-funds": "mutual-funds",
    "private-equity": "private-equity",
    "earnings": "earnings",
    "corporate-trends": "corporate-trends",
    "trade": "trade",
    "fta": "trade", # Alias for trade
    "foreign-trade": "trade",
    "indicators": "indicators",

    # Industry & Business
    "industry": "industry",
    "auto": "auto",
    "retail": "retail",
    "real-estate": "real-estate",
    "energy": "energy",
    "telecom": "telecom",
    "manufacturing": "manufacturing",
    "aviation": "aviation",
    "transportation": "transportation",
    "services": "services",
    "sme": "sme",

    # Technology & Science
    "technology": "technology",
    "tech": "technology",
    "ai": "ai",
    "startups": "startups",
    "science": "science",
    "isro": "space",
    "space": "space",
    "biotechnology": "biotechnology",
    "deeptech": "deeptech",
    "gadgets": "gadgets",
    "internet": "internet",
    "software": "software",
    "robotaxi": "automotive-tech",
    "starlink": "satellite-internet",
    "tiktok": "social-media",

    # Politics & Governance
    "politics": "politics",
    "politics-and-nation": "politics",
    "elections": "elections",
    "government": "governance",
    "judiciary": "judiciary",
    "parliament": "parliament",

    # World & International News
    "world": "world-news",
    "international": "world-news",
    "us": "us-news",
    "uk": "uk-news",
    "europe": "europe-news",
    "asia": "asia-news",
    "africa": "africa-news",
    "middle-east": "middle-east-news",
    "global-trends": "global-trends",

    # India Specific
    "india": "india",
    "delhi": "india-regional",
    "mumbai": "india-regional",
    "kolkata": "india-regional",
    "assam": "india-regional",
    "manipur": "india-regional",
    "nri": "nri",

    # Social & Lifestyle
    "wealth": "wealth",
    "education": "education",
    "career": "career",
    "jobs": "career",
    "opinion": "opinion",
    "panache": "lifestyle",
    "lifestyle": "lifestyle",
    "health": "health",
    "crime": "crime",
    "astrology": "astrology",
    "entertainment": "entertainment",
    "culture": "culture",
    "food": "food",

    # Environment & Climate
    "environment": "environment",
    "global-warming": "climate-change",
    "pollution": "pollution",
    "flora-fauna": "wildlife",
    "climate": "climate-change",
    "sustainability": "sustainability",

    # Defence & Security
    "defence": "defence",
    "military": "military",
    "security": "security",
    "terrorism": "terrorism",

    # Sports
    "sports": "sports",
    "cricket": "cricket",
    "football": "football",
    "hockey": "hockey",
    "chess": "chess",
    "olympics": "olympics",
    "wwe": "wrestling",
    "tour-de-france": "cycling",

    # Explainer/Analysis
    "et-explains": "explainer",
    "explainer": "explainer",
    "analysis": "analysis",

    # Uncategorized/General
    "new-updates": "general-updates",
    "general": "general"
}

BROADER_CATEGORY_RELATIONS = {
    "policy": "economy",
    "banking": "finance",
    "stocks": "markets",
    "ipos": "markets",
    "bonds": "markets",
    "forex": "finance",
    "mutual-funds": "finance",
    "private-equity": "finance",
    "earnings": "markets",
    "corporate-trends": "industry",
    "trade": "economy",
    "indicators": "economy",

    "auto": "industry",
    "retail": "industry",
    "real-estate": "industry",
    "energy": "industry",
    "telecom": "industry",
    "manufacturing": "industry",
    "aviation": "industry",
    "transportation": "industry",
    "services": "industry",

    "ai": "technology",
    "startups": "startup",
    "startup": "startup",
    "space": "science",
    "biotechnology": "science",
    "deeptech": "technology",
    "gadgets": "technology",
    "internet": "technology",
    "software": "technology",
    "automotive-tech": "technology",
    "satellite-internet": "technology",
    "social-media": "technology",

    "elections": "politics",
    "governance": "politics",
    "judiciary": "politics",
    "parliament": "politics",

    "us-news": "world-news",
    "uk-news": "world-news",
    "europe-news": "world-news",
    "asia-news": "world-news",
    "africa-news": "world-news",
    "middle-east-news": "world-news",
    "global-trends": "world-news",

    "india-regional": "india",

    "wealth": "lifestyle",
    "education": "social",
    "career": "social",
    "health": "social",
    "crime": "social",
    "astrology": "lifestyle",
    "entertainment": "lifestyle",
    "culture": "lifestyle",
    "food": "lifestyle",

    "climate-change": "environment",
    "pollution": "environment",
    "wildlife": "environment",
    "sustainability": "environment",

    "military": "defence",
    "security": "defence",
    "terrorism": "defence",

    "cricket": "sports",
    "football": "sports",
    "hockey": "sports",
    "chess": "sports",
    "olympics": "sports",
    "wrestling": "sports",
    "cycling": "sports",

    "general-updates": "general"
}

# === Helpers ===
def detect_categories(href: str, headline: str) -> List[str]:
    segments = urlparse(href).path.lower().split("/")
    domain_parts = urlparse(href).netloc.lower().split(".")
    combined = " ".join(segments + [headline.lower()] + domain_parts)

    detected_categories_set = set()

    for key_phrase, specific_category in CATEGORY_MAP.items():
        if re.search(r'\b' + re.escape(key_phrase.replace("-", "")) + r'\b', combined.replace("-", "")):
            detected_categories_set.add(specific_category)
            broader_cat = BROADER_CATEGORY_RELATIONS.get(specific_category)
            if broader_cat:
                detected_categories_set.add(broader_cat)

    return list(detected_categories_set or ["general"])


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
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)

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
            if res.status_code == 404:
                return
            elif res.status_code != 200:
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

            if timestamp:
                pub_time = datetime.strptime(timestamp[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            else:
                pub_time = datetime.now(timezone.utc)

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
                "timestamp": pub_time.isoformat().replace("+00:00", "Z")  # ✅ UTC ISO 8601
            }

            if not await collection.find_one({"ref": href}):
                await collection.insert_one(doc)
                print("✅ Inserted:", doc)

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
    now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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


@app.get("/all-refs", response_model=List[str])
async def get_all_refs():
    cursor = collection.find({}, {"_id": 0, "ref": 1})
    refs = []
    async for doc in cursor:
        refs.append(doc["ref"])
    return JSONResponse(content=refs)


@app.post("/subscribe")
async def subscribe(subscriber: Subscriber):
    try:
        existing_subscriber = await subscribers_collection.find_one({"email": subscriber.email})
        if existing_subscriber:
            return JSONResponse(
                status_code=409,
                content={"success": False, "message": "Email is already subscribed."}
            )

        await subscribers_collection.insert_one({
            "email": subscriber.email,
            "subscribed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        })
        print(f"New subscriber added: {subscriber.email}")
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "Subscription successful!"}
        )

    except Exception as e:
        print(f"An error occurred during subscription: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "An internal server error occurred."}
        )
