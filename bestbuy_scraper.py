#!/usr/bin/env python3
"""Scrape Best Buy Bluetooth Speaker reviews via hidden API with curl_cffi TLS impersonation.

This scraper bypasses Akamai bot protection by using curl_cffi's Chrome TLS
fingerprint impersonation and Best Buy's internal /ugc/v2/reviews JSON endpoint.
No proxies needed — TLS impersonation alone is sufficient for this endpoint.

Usage:
    python bestbuy_scraper.py                           # Full scrape
    python bestbuy_scraper.py --dry-run                  # Count reviews per product, skip save
    python bestbuy_scraper.py --max-products 10          # First 10 products only
    python bestbuy_scraper.py --reviews-per-product 50   # 50 reviews per product
"""

from curl_cffi import requests
import time
import random
import json
import csv
import os
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = "data_asg/bestbuy"
PRODUCTS_DIR = f"{OUTPUT_DIR}/products"
REVIEWS_PER_PRODUCT = 100
MAX_REVIEW_PAGES = 100       # safety limit (API also has its own pagination)
PAUSE_MIN = 0.5              # seconds between pagination requests
PAUSE_MAX = 1.5
REQUEST_TIMEOUT = 30

# Header for the hidden API call — Referer is required
API_HEADERS = {
    "Accept": "application/json",
    "Referer": "https://www.bestbuy.com/",
}

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

_session = requests.Session(impersonate="chrome124")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _pause() -> None:
    """Randomized delay between requests to avoid rate limiting."""
    time.sleep(random.uniform(PAUSE_MIN, PAUSE_MAX))


def fetch_reviews_page(sku: str, page: int) -> dict | None:
    """Fetch one page of reviews from Best Buy's hidden JSON API.

    Args:
        sku: Best Buy product SKU (numeric string).
        page: 1-indexed page number.

    Returns:
        Parsed JSON dict on success, None on failure.
    """
    url = (
        f"https://www.bestbuy.com/ugc/v2/reviews"
        f"?page={page}&pageSize=20&sku={sku}&sort=MOST_RECENT"
    )

    _pause()
    try:
        resp = _session.get(url, headers=API_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        print(
            f"  [ERROR] API returned HTTP {resp.status_code} "
            f"for SKU {sku} page {page}",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"  [ERROR] Request failed for SKU {sku} page {page}: {e}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Review extraction
# ---------------------------------------------------------------------------

def _parse_topic(topic: dict) -> dict:
    """Convert a raw API topic dict into a uniform review record."""
    return {
        "title": topic.get("title", "") or "",
        "text": topic.get("text", "") or "",
        "rating": topic.get("rating"),
        "date": topic.get("submissionTime", "") or "",
        "author": topic.get("author", "") or "",
    }


def scrape_product_reviews(
    sku: str,
    max_reviews: int = REVIEWS_PER_PRODUCT,
) -> list[dict]:
    """Scrape reviews for a single product via paginated API calls.

    Args:
        sku: Best Buy product SKU.
        max_reviews: Stop after collecting this many reviews.

    Returns:
        List of review dicts sorted by MOST_RECENT.
    """
    reviews: list[dict] = []

    for page in range(1, MAX_REVIEW_PAGES + 1):
        if len(reviews) >= max_reviews:
            break

        data = fetch_reviews_page(sku, page)
        if not data:
            break

        topics = data.get("topics") or []
        if not topics:
            break

        for topic in topics:
            if topic.get("topicType") == "review":
                reviews.append(_parse_topic(topic))

        total_pages = data.get("totalPages", 0)
        if page >= total_pages:
            break

    return reviews[:max_reviews]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_product_data(
    sku: str,
    name: str,
    reviews: list[dict],
    output_dir: str = PRODUCTS_DIR,
) -> None:
    """Save per-product JSON with metadata and reviews."""
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/{sku}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"sku": sku, "name": name, "reviews": reviews},
                f,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as e:
        print(f"  [ERROR] Could not write {path}: {e}", file=sys.stderr)


def save_consolidated(
    products_data: list[dict],
    reviews_data: list[dict],
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Save consolidated products.json and all_reviews.csv."""
    os.makedirs(output_dir, exist_ok=True)

    # Products catalog
    catalog_path = f"{output_dir}/products.json"
    catalog = [
        {
            "sku": p["sku"],
            "name": p["name"],
            "review_count": p["review_count"],
        }
        for p in products_data
    ]
    try:
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"  Products catalog: {catalog_path}")
    except OSError as e:
        print(f"  [ERROR] Could not write {catalog_path}: {e}", file=sys.stderr)

    # Consolidated CSV — same schema as before
    csv_path = f"{output_dir}/all_reviews.csv"
    fieldnames = ["product_name", "review_text", "rating", "date"]
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in reviews_data:
                writer.writerow(row)
        print(f"  Reviews CSV: {csv_path}")
    except OSError as e:
        print(f"  [ERROR] Could not write {csv_path}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_product_list(path: str | None = None) -> list[dict]:
    """Load the curated product list from a JSON file.

    Defaults to ``data_asg/bestbuy/products.json``.
    Each entry must have at least ``sku`` and ``name``.
    """
    if path is None:
        path = f"{OUTPUT_DIR}/products.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            products = json.load(f)
        if not products:
            print(f"[ERROR] Product list is empty: {path}", file=sys.stderr)
            sys.exit(1)
        return products
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Could not load product list from {path}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Orchestrate the full scraping pipeline."""
    # ---- CLI overrides ----
    dry_run = "--dry-run" in sys.argv
    max_products = None
    reviews_per_product = REVIEWS_PER_PRODUCT

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--max-products" and i + 1 < len(args):
            max_products = int(args[i + 1])
        elif arg == "--reviews-per-product" and i + 1 < len(args):
            reviews_per_product = int(args[i + 1])

    # ---- Header ----
    print("=" * 60)
    print("  Best Buy Speaker Review Scraper  (curl_cffi + Hidden API)")
    print("=" * 60)
    print(f"  Dry run:    {'YES' if dry_run else 'no'}")
    print(f"  Reviews/product: {reviews_per_product}")
    print()

    # ---- Step 1: Load product list ----
    products = load_product_list()
    if max_products:
        products = products[:max_products]
    print(f"[1/3] Loaded {len(products)} products from product list")
    print()

    # ---- Step 2: Scrape reviews ----
    print(f"[2/3] Scraping reviews via hidden API...")
    print()
    all_reviews: list[dict] = []
    products_with_data: list[dict] = []

    for i, product in enumerate(products, 1):
        sku = product["sku"]
        name = product["name"]
        print(f"  ({i}/{len(products)}) [{sku}] {name}")

        # Pre-query total_pages for progress indication (quick page=1 call)
        preview = fetch_reviews_page(sku, 1)
        if preview:
            total = preview.get("totalPages", "?")
            print(f"         API reports ~{total} pages available")
        else:
            print(f"         ⚠ Could not reach API — skipping")
            continue

        reviews = scrape_product_reviews(sku, reviews_per_product)
        print(f"         → {len(reviews)} reviews collected")

        if not dry_run:
            save_product_data(sku, name, reviews)

        for r in reviews:
            all_reviews.append({
                "product_name": name,
                "review_text": r["text"],
                "rating": r["rating"],
                "date": r["date"],
            })

        products_with_data.append({**product, "review_count": len(reviews)})
        print()

    # ---- Step 3: Save consolidated output ----
    if dry_run:
        print(f"[3/3] DRY RUN — no files written")
    else:
        print(f"[3/3] Saving consolidated output...")
        save_consolidated(products_with_data, all_reviews)

    print()
    print("=" * 60)
    print("  SCRAPE COMPLETE")
    print("=" * 60)
    print(f"  Products scraped:   {len(products_with_data)}")
    print(f"  Total reviews:      {len(all_reviews)}")
    print(f"  Output directory:   {OUTPUT_DIR}/")
    print()

    # Warn about low-review products
    low = [(p["name"], p["review_count"]) for p in products_with_data
           if p["review_count"] < 80]
    if low:
        print(f"  ⚠ Products with < 80 reviews ({len(low)}):")
        for name, count in low:
            print(f"    - {name}: {count} reviews")
        print()

    print("  Next: run aspect_extraction.ipynb for topic modeling.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Scraping stopped by user. Partial data may have been saved.")
        sys.exit(1)
