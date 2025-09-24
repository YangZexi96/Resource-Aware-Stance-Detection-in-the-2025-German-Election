#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# ─── Bootstrap src/ into import path ────────────────────────────────────────────
HERE      = Path(__file__).resolve()
PROJECT   = HERE.parents[2]        # …/src/0_scraping_pipeline → MasterThesis_final
SRC_DIR   = PROJECT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import json
import asyncio
import csv
import random
import calendar
import re
from datetime import date, timedelta
from typing import Dict, Any, List

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError
from loguru import logger as log

from utils.config import BASE_DIR, LIST_DIR, TWEETS_DIR, NITTER_BASE

# ─── Load target accounts from list CSVs ────────────────────────────────────────
TOP10_CSV = os.path.join(LIST_DIR, 'top10_per_party.csv')
TOP3_CSV  = os.path.join(LIST_DIR, 'top3_per_party.csv')

df10 = pd.read_csv(TOP10_CSV, sep=';', engine='python')
df3  = pd.read_csv(TOP3_CSV,  sep=';', engine='python')

# Combine handles preserving order, remove duplicates
accounts_10 = df10['handle'].tolist()
accounts_3  = df3['handle'].tolist()
ACCOUNTS    = list(dict.fromkeys(accounts_10 + accounts_3))

# Build handle→party mapping
mapping_df = pd.concat([
    df10[['handle', 'T_Partei']],
    df3[['handle', 'T_Partei']]
], ignore_index=True)
CANDIDATE_TO_PARTY = dict(zip(mapping_df['handle'], mapping_df['T_Partei']))

# ─── Directories & Progress file ──────────────────────────────────────────────
RAW_AUGMENT_DIR = os.path.join(TWEETS_DIR, 'augmentation_dataset')
PROGRESS_FILE   = os.path.join(RAW_AUGMENT_DIR, 'progress.json')

# ─── Scraper config ────────────────────────────────────────────────────────────
MAX_RETRIES = 6
BATCH_SIZE  = 200

# Hardcoded scrape dates
SCRAPE_DATES = [
    date(2024, 9, 2),
    date(2024, 10, 17),
    date(2024, 11, 7),
    date(2024, 12, 18),
    date(2025, 1, 23),
    date(2025, 2, 23),
    date(2025, 3, 25),
]

# ─── Progress helpers ──────────────────────────────────────────────────────────
def load_progress() -> Dict[str, Any]:
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_progress(progress: Dict[str, Any]):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)

def get_day_state(prog, acct, day_str):
    acct_prog = prog.setdefault(acct, {})
    return acct_prog.setdefault(day_str, {"done": False, "cursor": None})

# ─── URL builder ───────────────────────────────────────────────────────────────
def build_search_url(base: str, user: str, since: date, until: date) -> str:
    return (
        f"{base}/search?f=tweets"
        f"&q=%40{user}"
        f"&e-media=on"
        f"&e-images=on"
        f"&e-videos=on"
        f"&e-native_video=on"
        f"&since={since.isoformat()}"
        f"&until={until.isoformat()}"
    )

def extract_since_from_url(url: str) -> str:
    m = re.search(r'since=(\d{4}-\d{2}-\d{2})', url)
    return m.group(1) if m else None

# ─── Tweet extraction ───────────────────────────────────────────────────────────
async def extract_tweet(tweet):
    for attempt in range(MAX_RETRIES):
        try:
            await tweet.wait_for(timeout=5000)
            if (await tweet.locator(".tweet-content").count() == 0
                and await tweet.locator(".quote-text").count() == 0
                and await tweet.locator(".replying-to").count() == 0):
                return None

            text = ""
            if await tweet.locator(".tweet-content").count() > 0:
                text = (await tweet.locator(".tweet-content").nth(0)
                              .inner_text()).strip()
            if len(text) < 5:
                return None

            username = await tweet.locator(
                ".fullname-and-username a.username"
            ).nth(0).inner_text()
            is_verified = await tweet.locator(".verified-icon").count() > 0
            vtype = "None"
            if await tweet.locator(".verified-icon.blue").count():   vtype = "Blue"
            elif await tweet.locator(".verified-icon.gold").count(): vtype = "Gold"
            elif await tweet.locator(".verified-icon.gray").count(): vtype = "Gray"

            retu = await tweet.locator(".retweet-header").count()
            retweeted = await tweet.locator(
                ".retweet-header"
            ).inner_text() if retu else None
            ttype = f"Retweet by {retweeted}" if retweeted else "Original"

            hashtags = await tweet.locator("a").evaluate_all(
                "els=>els.map(e=>e.textContent||'')"
                ".filter(t=>t.startsWith('#')).map(t=>t.trim())"
            )
            mentions = await tweet.locator("a").evaluate_all(
                "els=>els.map(e=>e.textContent||'')"
                ".filter(t=>t.startsWith('@')).map(t=>t.trim())"
            )
            urls = await tweet.locator("a").evaluate_all(
                "els=>els.map(e=>e.getAttribute('href')||'')"
                ".filter(h=>h.startsWith('http'))"
            )
            stats = await tweet.locator(
                ".tweet-stats span.tweet-stat"
            ).evaluate_all(
                "els=>els.map(e=>e.innerText.trim())"
                ".filter(t=>/^\\d+$/.test(t))"
            )
            comments, retweets, quotes, likes = (stats + ["0"]*4)[:4]
            timestamp = None
            if await tweet.locator(".tweet-date a").count():
                timestamp = await tweet.locator(
                    ".tweet-date a"
                ).nth(0).get_attribute('title')

            is_reply = await tweet.locator(".replying-to").count() > 0
            replies = []
            if is_reply:
                replies = await tweet.locator(".replying-to a")\
                                      .evaluate_all("els=>els.map(e=>e.textContent.trim())")

            quote_text = None
            quote_author = None
            if await tweet.locator(".quote-text").count():
                quote_text = await tweet.locator(".quote-text")\
                                        .nth(0).inner_text()
            if await tweet.locator(
                ".quote .fullname-and-username a.username"
            ).count():
                quote_author = await tweet.locator(
                    ".quote .fullname-and-username a.username"
                ).nth(0).inner_text()

            return {
                'username': username,
                'verified': is_verified,
                'verified_type': vtype,
                'tweet_text': text,
                'tweet_type': ttype,
                'hashtags': '; '.join(hashtags),
                'mentions': '; '.join(mentions),
                'urls': '; '.join(urls),
                'comments': comments,
                'retweets': retweets,
                'quotes': quotes,
                'likes': likes,
                'timestamp': timestamp,
                'is_reply': is_reply,
                'reply_to_usernames': '; '.join(replies),
                'quoted_tweet_text': quote_text,
                'quoted_tweet_author': quote_author,
                'party': CANDIDATE_TO_PARTY.get(username)
            }
        except TimeoutError:
            log.warning("Tweet extraction timeout, retrying")
            await asyncio.sleep(1)
    return None

async def save_tweets_to_csv(tweets: List[Dict], fname: str):
    if not tweets:
        return
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fieldnames = list(tweets[0].keys())
    exists = os.path.exists(fname)
    with open(fname, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL
        )
        if not exists:
            writer.writeheader()
        writer.writerows(tweets)
    log.info(f"Saved {len(tweets)} tweets to {fname}")

async def scrape_nitter_mentions(ctx, user, url, base, csv_file):
    prog    = load_progress()
    day_str = extract_since_from_url(url)
    state   = get_day_state(prog, user, day_str)

    log.info(f"Scraping @{user} on {day_str}")
    page = await ctx.new_page()
    await ctx.route('**/*',
        lambda r: r.abort() if r.request.resource_type in
                         ['image','stylesheet','font','media']
                     else r.continue_()
    )
    buffer = []
    fails  = 0

    try:
        while True:
            try:
                await page.goto(url, timeout=15000, wait_until='domcontentloaded')
            except TimeoutError:
                log.error(f"Page.goto timeout on {url}; retrying")
                fails += 1
                if fails >= MAX_RETRIES:
                    state["cursor"] = url
                    save_progress(prog)
                    raise
                await asyncio.sleep(10)
                continue

            state["cursor"] = url
            save_progress(prog)

            try:
                await page.wait_for_selector('.timeline-item', timeout=5000)
            except TimeoutError:
                log.error("No tweets loaded, retrying navigation")
                fails += 1
                if fails >= MAX_RETRIES:
                    state["cursor"] = url
                    save_progress(prog)
                    raise
                await asyncio.sleep(10)
                continue
            fails = 0

            items = await page.locator('.timeline-item').all()
            for it in items:
                d = await extract_tweet(it)
                if d:
                    buffer.append(d)
                if len(buffer) >= BATCH_SIZE:
                    await save_tweets_to_csv(buffer, csv_file)
                    buffer.clear()
                    state["cursor"] = url
                    save_progress(prog)

            more = await page.locator('.show-more a').all()
            if not more:
                break
            href = await more[-1].get_attribute('href')
            if 'cursor' not in href:
                break
            url = (base + href) if href.startswith('/') else base + '/search' + href

        if buffer:
            await save_tweets_to_csv(buffer, csv_file)

        state["done"] = True
        save_progress(prog)
    finally:
        await page.close()

async def orchestrator():
    prog = load_progress()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx     = await browser.new_context()

        for acct in ACCOUNTS:
            for day in SCRAPE_DATES:
                ds    = day.isoformat()
                state = get_day_state(prog, acct, ds)
                if state["done"]:
                    continue

                url      = state["cursor"] or build_search_url(
                    NITTER_BASE, acct, day, day + timedelta(days=1)
                )
                csv_file = os.path.join(RAW_AUGMENT_DIR, acct, f"{ds}_{acct}.csv")

                try:
                    await scrape_nitter_mentions(
                        ctx, acct, url, NITTER_BASE, csv_file
                    )
                except Exception as e:
                    log.error(f"Scraping @{acct} on {ds} failed: {e!r}")
                    continue

        await browser.close()

async def main():
    log.info("Starting augmentation scraper")
    await orchestrator()
    log.info("Augmentation scraping complete")

if __name__ == '__main__':
    asyncio.run(main())
