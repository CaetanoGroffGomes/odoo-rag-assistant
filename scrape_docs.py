# scrape_docs.py — crawler async com progresso, retries e checkpoint
import asyncio, json, os, re, signal, time, email.utils as eut
from dataclasses import dataclass
from urllib.parse import urljoin, urldefrag, urlparse

import aiohttp
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm
import urllib.robotparser as robotparser
from typing import Optional, Dict, Tuple, List


START_URL = "https://www.odoo.com/documentation/16.0/pt_BR/"
ALLOWED_HOST = "www.odoo.com"
HEADERS = {"User-Agent": "ERP-RAG/1.0 (+research; contact: you@example.com)"}

CHECKPOINT_EVERY = 50
DEFAULT_MAX_PAGES = 600
DEFAULT_CONCURRENCY = 10
OUT_PATH = "docs.json"
STATE_PATH = ".scrape_state.json"

MIN_TEXT_LEN = 300

@dataclass
class State:
    seen: set
    todo: list
    docs: list

def clean_url(u: str) -> str:
    u = urldefrag(u)[0]
    return u.strip()

def is_allowed(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and p.netloc == ALLOWED_HOST
    except Exception:
        return False

def extract(html: str):
    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main") or soup
    h = main.find(["h1", "h2"])
    title = h.get_text(" ", strip=True) if h else (soup.title.get_text(" ", strip=True) if soup.title else "")
    text = re.sub(r"[ \t]+", " ", main.get_text(" ", strip=True))
    return title, text, soup

async def fetch(session: aiohttp.ClientSession, url: str, *, retries: int = 3) -> str:
    """
    GET com SSL e HEADERS; retorna somente o HTML (ou "" em caso de erro).
    Mantém compatibilidade com o worker existente.
    """
    delay = 0.5
    for attempt in range(1, retries + 1):
        try:
            async with session.get(
                url,
                ssl=True,  # antes estava False
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as r:
                ctype = r.headers.get("Content-Type", "")
                if r.status == 200 and "text/html" in ctype:
                    return await r.text()
                # drena caso não seja HTML/200
                _ = await r.read()
        except Exception:
            pass
        await asyncio.sleep(delay)
        delay *= 1.7
    return ""



def save_checkpoint(state: State):
    tmp = {"seen": list(state.seen), "todo": state.todo, "docs": state.docs}
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(tmp, f, ensure_ascii=False)

def load_checkpoint() -> State | None:
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            tmp = json.load(f)
        return State(seen=set(tmp["seen"]), todo=tmp["todo"], docs=tmp["docs"])
    except Exception:
        return None

async def crawl(max_pages=DEFAULT_MAX_PAGES, concurrency=DEFAULT_CONCURRENCY, out_path=OUT_PATH, start_url=START_URL):
    st = load_checkpoint()
    if st is None:
        st = State(seen=set(), todo=[start_url], docs=[])

    sem = asyncio.Semaphore(concurrency)
    session_timeout = aiohttp.ClientTimeout(total=25)
    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300, ssl=True)

    start_t = time.time()
    scraped = 0

    async with aiohttp.ClientSession(headers=HEADERS, timeout=session_timeout, connector=connector) as session:
        pbar = tqdm(total=max_pages, desc="Scraping", unit="page", dynamic_ncols=True)
        pbar.update(len(st.docs) % max_pages)

        async def worker(url: str):
            nonlocal scraped
            async with sem:
                html = await fetch(session, url)
            if not html:
                return [], None

            title, text, soup = extract(html)
            if len(text) >= MIN_TEXT_LEN:
                st.docs.append({"title": title, "content": text, "url": url})
                scraped += 1
                pbar.update(1)

            new_links = []
            for a in soup.select("a[href]"):
                href = a.get("href", "")
                nxt = urljoin(url, href)
                nxt = clean_url(nxt)
                if is_allowed(nxt) and nxt not in st.seen:
                    new_links.append(nxt)
            return new_links, url

        while st.todo and len(st.docs) < max_pages:
            batch = []
            while st.todo and len(batch) < concurrency:
                u = st.todo.pop(0)
                if u in st.seen:
                    continue
                st.seen.add(u)
                batch.append(u)

            tasks = [asyncio.create_task(worker(u)) for u in batch]
            for coro in asyncio.as_completed(tasks):
                new_links, url_done = await coro
                if new_links:
                    st.todo.extend(new_links)

            if scraped and scraped % CHECKPOINT_EVERY == 0:
                save_checkpoint(st)

        pbar.close()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(st.docs, f, ensure_ascii=False, indent=2)

    if os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)

    dur = time.time() - start_t
    print(f"📦 Salvos {len(st.docs)} docs em {out_path} | ⏱ {dur:.1f}s | ~{(len(st.docs)/max(dur,1)):.2f} pág/s")

def _install_sigint_handler():
    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
        except NotImplementedError:
            pass

if __name__ == "__main__":
    import argparse
    _install_sigint_handler()

    ap = argparse.ArgumentParser()
    ap.add_argument("--start-url", default=START_URL)
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    try:
        asyncio.run(crawl(
            max_pages=args.max_pages,
            concurrency=args.concurrency,
            out_path=args.out,
            start_url=args.start_url
        ))
    except KeyboardInterrupt:
        print("\nInterrompido. Checkpoint salvo em", STATE_PATH)
