#!/usr/bin/env python3
"""Scrape Best Buy Bluetooth & Wireless Speaker reviews for aspect-based sentiment analysis."""

import requests
from bs4 import BeautifulSoup
import time
import random
import json
import csv
import os
import sys
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENTS = [
    # Windows Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Mac Chrome
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Linux Chrome
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]

CATEGORY_URL = (
    "https://www.bestbuy.com/site/portable-audio/portable-speakers-docks-radios/"
    "pcmcat310200050004.c?id=pcmcat310200050004&intl=nosplash"
)

OUTPUT_DIR = "data_asg/bestbuy"
PRODUCTS_DIR = f"{OUTPUT_DIR}/products"
MIN_REVIEWS = 80
MAX_PRODUCTS = 40
MAX_REVIEW_PAGES = 20
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

_session = requests.Session()


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> str:
    """Fetch URL with randomized delay and rotating user-agent."""
    delay = random.uniform(1.0, 3.0)
    time.sleep(delay)

    user_agent = random.choice(USER_AGENTS)
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = _session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Request failed for {url}: {e}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_listing(soup: BeautifulSoup) -> list[dict]:
    """Extract product cards from listing page HTML.

    Returns a list of dicts with keys: sku, name, price, rating.
    """
    products = []

    # Attempt 1: find product containers via common Best Buy patterns
    product_elements = (
        soup.find_all("li", class_=re.compile(r"sku-item", re.I))
        or soup.find_all("li", class_=re.compile(r"product-item", re.I))
        or soup.find_all("div", attrs={"data-product-sku": True})
    )

    for elem in product_elements:
        try:
            sku = (
                elem.get("data-sku-id")
                or elem.get("data-product-sku")
                or None
            )
            name_el = elem.find(["h4", "h3", "h2", "a"], class_=re.compile(r"(name|title|heading)", re.I))
            name = name_el.get_text(strip=True) if name_el else None

            price_el = elem.find("span", class_=re.compile(r"price", re.I))
            price = None
            if price_el:
                price_text = price_el.get_text(strip=True).replace("$", "").replace(",", "")
                try:
                    price = float(price_text)
                except (ValueError, TypeError):
                    price = None

            rating = None
            rating_el = elem.find("span", class_=re.compile(r"rating|star", re.I))
            if rating_el:
                rating_text = rating_el.get("data-rating") or rating_el.get_text(strip=True)
                try:
                    rating = float(rating_text)
                except (ValueError, TypeError):
                    rating = None

            products.append({
                "sku": sku,
                "name": name,
                "price": price,
                "rating": rating,
            })
        except Exception:
            continue

    # Fallback: parse JSON-LD embedded data
    if not products:
        script_tags = soup.find_all("script", type="application/ld+json")
        for script in script_tags:
            try:
                data = json.loads(script.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict) and item.get("@type") in ("Product", "ItemList"):
                        if item.get("@type") == "ItemList" and "itemListElement" in item:
                            for el in item["itemListElement"]:
                                prod = el.get("item", el)
                                products.append(_parse_ld_product(prod))
                        else:
                            products.append(_parse_ld_product(item))
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

    return products


def _parse_ld_product(item: dict) -> dict:
    """Parse a single product from JSON-LD structured data."""
    offers = item.get("offers", {})
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    price_raw = offers.get("price", None)
    try:
        price = float(price_raw) if price_raw else None
    except (ValueError, TypeError):
        price = None

    rating_raw = None
    agg_rating = item.get("aggregateRating", {})
    if agg_rating:
        rating_raw = agg_rating.get("ratingValue", None)
    try:
        rating = float(rating_raw) if rating_raw else None
    except (ValueError, TypeError):
        rating = None

    return {
        "sku": item.get("sku", item.get("mpn", None)),
        "name": item.get("name", None),
        "price": price,
        "rating": rating,
    }


def parse_reviews(soup: BeautifulSoup) -> list[dict]:
    """Extract review items from review page HTML.

    Returns a list of dicts with keys: title, text, rating, date.
    """
    reviews = []

    # Attempt 1: structural selectors
    review_elements = (
        soup.find_all("div", class_=re.compile(r"review-item", re.I))
        or soup.find_all("div", class_=re.compile(r"review-content", re.I))
        or soup.find_all("li", class_=re.compile(r"review", re.I))
    )

    for elem in review_elements:
        try:
            title_el = elem.find(["h4", "h3", "h2", "strong", "b"],
                                 class_=re.compile(r"(title|heading)", re.I))
            title = title_el.get_text(strip=True) if title_el else None

            text_el = elem.find("p", class_=re.compile(r"(review|text|content|body)", re.I))
            text = text_el.get_text(strip=True) if text_el else None

            rating_el = elem.find(class_=re.compile(r"(rating|star)", re.I))
            rating = None
            if rating_el:
                # Try data attribute first, then class name with number, then text
                rating_str = (
                    rating_el.get("data-rating")
                    or rating_el.get("data-score")
                    or _extract_rating_from_class(rating_el.get("class", []))
                    or rating_el.get_text(strip=True)
                )
                try:
                    rating = float(rating_str)
                except (ValueError, TypeError):
                    rating = None

            date_el = (
                elem.find("time")
                or elem.find("span", class_=re.compile(r"(date|time|submission)", re.I))
                or elem.find("span", string=re.compile(r"\d{4}"))
            )
            date = date_el.get("datetime") or date_el.get_text(strip=True) if date_el else None

            reviews.append({
                "title": title or "",
                "text": text or "",
                "rating": rating,
                "date": date or "",
            })
        except Exception:
            continue

    # Attempt 2: JSON-LD fallback for review data
    if not reviews:
        script_tags = soup.find_all("script", type="application/ld+json")
        for script in script_tags:
            try:
                data = json.loads(script.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict) and item.get("@type") == "Review":
                        reviews.append(_parse_ld_review(item))
                    elif isinstance(item, dict) and "review" in item:
                        r = item["review"]
                        if isinstance(r, list):
                            for rv in r:
                                reviews.append(_parse_ld_review(rv))
                        else:
                            reviews.append(_parse_ld_review(r))
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

    return reviews


def _extract_rating_from_class(classes) -> float | None:
    """Extract rating number from class names like 'rating-4' or 'stars-5'."""
    if not classes:
        return None
    for cls in classes:
        m = re.search(r"(\d+(?:\.\d+)?)", cls)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _parse_ld_review(item: dict) -> dict:
    """Parse a single review from JSON-LD structured data."""
    rating_raw = None
    review_rating = item.get("reviewRating", {})
    if review_rating:
        rating_raw = review_rating.get("ratingValue", None)
    try:
        rating = float(rating_raw) if rating_raw else None
    except (ValueError, TypeError):
        rating = None

    return {
        "title": item.get("name", ""),
        "text": item.get("reviewBody", ""),
        "rating": rating,
        "date": item.get("datePublished", ""),
    }


# ---------------------------------------------------------------------------
# Scraping orchestration
# ---------------------------------------------------------------------------

def scrape_product_listing(category_url: str, max_products: int = MAX_PRODUCTS) -> list[dict]:
    """Scrape product listing pages and return product metadata."""
    products = []
    page_url = category_url

    while len(products) < max_products:
        print(f"  Fetching product listing: {page_url[:80]}...")
        html = fetch_page(page_url)
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")
        page_products = parse_listing(soup)

        # Deduplicate by SKU
        existing_skus = {p["sku"] for p in products if p["sku"]}
        for p in page_products:
            if p["sku"] and p["sku"] not in existing_skus:
                products.append(p)
                existing_skus.add(p["sku"])

        if len(page_products) == 0:
            break  # No more products found

        # Try next page
        next_link = soup.find("a", class_=re.compile(r"next", re.I)) or \
                    soup.find("a", string=re.compile(r"next", re.I)) or \
                    soup.find("link", attrs={"rel": "next"})
        if next_link:
            href = next_link.get("href")
            if href:
                page_url = href if href.startswith("http") else f"https://www.bestbuy.com{href}"
            else:
                break
        else:
            # Try pagination URL addition
            if "?page=" in page_url:
                current_page = int(re.search(r"page=(\d+)", page_url).group(1))
                page_url = re.sub(r"page=\d+", f"page={current_page + 1}", page_url)
            else:
                page_url = f"{category_url}&page=2"

    return products[:max_products]


def scrape_reviews(sku: str, product_name: str, max_reviews: int = 100) -> list[dict]:
    """Scrape review pages for a product and return a list of reviews."""
    all_reviews = []

    for page in range(1, MAX_REVIEW_PAGES + 1):
        slug = product_name.lower().replace(" ", "-").replace("/", "-")
        url = f"https://www.bestbuy.com/site/reviews/{slug}/{sku}?page={page}"
        print(f"    Page {page}...")
        html = fetch_page(url)
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")
        reviews = parse_reviews(soup)

        if not reviews:
            break  # No more reviews

        all_reviews.extend(reviews)

        if len(all_reviews) >= max_reviews:
            all_reviews = all_reviews[:max_reviews]
            break

    return all_reviews


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_product_data(product: dict, reviews: list[dict], output_dir: str = PRODUCTS_DIR) -> None:
    """Save per-product JSON with metadata and all reviews."""
    os.makedirs(output_dir, exist_ok=True)
    output = {**product, "reviews": reviews, "total_reviews_collected": len(reviews)}
    path = f"{output_dir}/{product['sku']}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"  [ERROR] Could not write {path}: {e}", file=sys.stderr)


def save_consolidated(products_data: list[dict], reviews_data: list[dict],
                      output_dir: str = OUTPUT_DIR) -> None:
    """Save consolidated products.json and all_reviews.csv."""
    os.makedirs(output_dir, exist_ok=True)

    # Products catalog (metadata only, no reviews)
    catalog_path = f"{output_dir}/products.json"
    catalog = [{"sku": p["sku"], "name": p["name"], "price": p["price"], "rating": p["rating"]}
               for p in products_data]
    try:
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"  Products catalog: {catalog_path}")
    except OSError as e:
        print(f"  [ERROR] Could not write {catalog_path}: {e}", file=sys.stderr)

    # Consolidated CSV
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

def main() -> None:
    """Orchestrate the full scraping pipeline."""
    print("=" * 60)
    print("  Best Buy Speaker Review Scraper")
    print("=" * 60)
    print()

    os.makedirs(PRODUCTS_DIR, exist_ok=True)

    # Step 1: scrape product listing
    print("[1/3] Scraping product listing...")
    products = scrape_product_listing(CATEGORY_URL, MAX_PRODUCTS)
    print(f"  Found {len(products)} products")
    print()

    if not products:
        print("[ERROR] No products found. Check the category URL.")
        sys.exit(1)

    # Step 2: scrape reviews for each product
    print(f"[2/3] Scraping reviews for {len(products)} products...")
    print()
    all_reviews = []
    products_with_data = []

    for i, product in enumerate(products, 1):
        name = product["name"] or f"SKU-{product['sku']}"
        print(f"  ({i}/{len(products)}) {name}")
        reviews = scrape_reviews(product["sku"], name)
        save_product_data(product, reviews)
        print(f"    → {len(reviews)} reviews collected")

        for r in reviews:
            all_reviews.append({
                "product_name": name,
                "review_text": r["text"],
                "rating": r["rating"],
                "date": r["date"],
            })

        products_with_data.append({**product, "review_count": len(reviews)})
        print()

    # Step 3: save consolidated output
    print(f"[3/3] Saving consolidated output...")
    save_consolidated(products_with_data, all_reviews)
    print()

    # Final summary
    print("=" * 60)
    print("  SCRAPE COMPLETE")
    print("=" * 60)
    print(f"  Total products scraped: {len(products_with_data)}")
    print(f"  Total reviews collected: {len(all_reviews)}")
    print(f"  Output: {OUTPUT_DIR}/")
    print()

    # Warn about products with fewer than 80 reviews
    low = [(p["name"], p["review_count"]) for p in products_with_data
           if p["review_count"] < MIN_REVIEWS]
    if low:
        print(f"  ⚠ Products with < {MIN_REVIEWS} reviews ({len(low)}):")
        for name, count in low:
            print(f"    - {name}: {count} reviews")

    print()
    print("  Next: load data into aspect_extraction.ipynb for topic modeling.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Scraping stopped by user. Partial data may have been saved.")
        sys.exit(1)
