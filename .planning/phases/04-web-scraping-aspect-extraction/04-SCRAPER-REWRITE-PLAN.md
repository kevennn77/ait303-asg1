# Scraper Rewrite Plan — curl_cffi + Hidden Best Buy API

## Problem

The current `bestbuy_scraper.py` uses `requests.Session()` + `BeautifulSoup` to parse Best Buy's HTML pages. This fails because Best Buy uses **Akamai Bot Manager** which:

1. Blocks all non-browser HTTP clients (Python `requests`, `httpx`, etc.)
2. Uses TLS fingerprinting to detect headless browsers (Playwright, Selenium)
3. Only passes `curl_cffi` with Chrome124+ TLS impersonation

We confirmed: `curl_cffi.Session(impersonate='chrome124')` reaches Best Buy's hidden JSON API successfully (Status 200, full responses returned).

---

## Solution: curl_cffi + Hidden Reviews API

### Hidden API Endpoint

```
GET https://www.bestbuy.com/ugc/v2/reviews
  ?sku={SKU}
  &page={N}
  &pageSize=20
  &sort=MOST_RECENT
```

**Response shape (confirmed):**
```json
{
  "totalPages": 332,
  "topics": [
    {
      "rating": 5,
      "title": "Great speaker",
      "text": "Amazing sound quality...",
      "author": "Username",
      "submissionTime": "2024-03-02T10:52:07.000-06:00",
      "positiveFeedbackCount": 3,
      "negativeFeedbackCount": 0
    }
  ]
}
```

### Product Discovery via Sitemaps

All 9 product sitemaps (`sitemap_p_0` through `sitemap_p_8.xml.gz`) parsed. Filtered for Bluetooth speakers, validated via live API.

**Result:** 214 unique products with 100+ reviews confirmed across major brands:

| Brand | Count |
|-------|-------|
| Altec Lansing | 25+ models (Baby Boom, Boom Jacket, Life Jacket, Mini H2O) |
| Anker Soundcore | 8 models (Flare, Select, Mini, Motion Boom) |
| JBL | 6 models (Charge, Flip, Clip, Go, PartyBox, Xtreme) |
| Bose | 6 models (SoundLink Flex, Micro, Revolve, Portable Smart) |
| Sony | 6 models (SRS-XB13 through XG300) |
| Beats | 4 models (Pill, Pill+, Beatbox) |
| Ultimate Ears | 4 models (BOOM 3, MEGABOOM 3, Wonderboom) |
| Marshall | 4 models (Emberton, Stanmore, Kilburn, Acton) |
| Braven | 6 models (BRV-1, BRV-HD, Balance, XXL) |
| Bang & Olufsen | 3 models (Beolit 20, Beosound A1, Beosound Explore) |
| Others (808, Aiwa, Harman Kardon, etc.) | 5+ |

---

## File Changes

### 1. `bestbuy_scraper.py` — Full Rewrite

**Current (487 lines):** `requests` + `BeautifulSoup` for HTML parsing, category page crawling, review HTML parsing.

**New (~350 lines):** `curl_cffi` + direct JSON API, product list from curated file, review JSON parsing.

| Function | Current | New |
|----------|---------|-----|
| `fetch_page()` | `requests.get()` with UA rotation + BS4 | `session.get()` with impersonation, JSON parsing |
| `parse_listing()` | BS4 category page parsing | **REMOVED** — products from curated list |
| `parse_reviews()` | BS4 review HTML parsing | **REMOVED** — JSON directly from API |
| `scrape_product_listing()` | Paginate category pages | **REMOVED** — products from curated list |
| `scrape_reviews()` | Build review page URL → parse HTML | Hit `/ugc/v2/reviews?sku={SKU}` → parse JSON |
| `main()` | Step 1: category crawl → Step 2: reviews | Load product list → iterate → rate limit → save |

**New functions:**
- `load_product_list(path)` — Load curated product list from JSON
- `fetch_reviews_api(sku, page, session)` — Single API call with curl_cffi impersonation
- `scrape_product_reviews(sku, name, session, max_reviews)` — Paginate through API until max_reviews
- All existing output functions (`save_product_data`, `save_consolidated`) remain **unchanged**

### 2. `tests/test_scraper.py` — Rewrite

**Current:** Uses BS4 mocks, tests HTML parsing logic (now irrelevant).

**New tests:**
| Test | Description |
|------|-------------|
| `test_curl_cffi_session_creation` | Verify `curl_cffi.Session(impersonate='chrome124')` can be created |
| `test_fetch_reviews_api_response_format` | Mock API response → verify JSON parsing into review dicts |
| `test_scrape_product_reviews_pagination` | Mock paginated API → verify review accumulation |
| `test_api_schema_evolution` | Verify all JSON fields we depend on exist in response |
| `test_csv_schema_unchanged` | **SAME** — verify `all_reviews.csv` columns (product_name, review_text, rating, date) |
| `test_no_pii_in_scraped_data` | **SAME** — no PII regex check |

### 3. `tests/conftest.py` — Add fixtures

Add `mock_api_response` fixture returning a sample `/ugc/v2/reviews` JSON payload.

### 4. `notebooks/aspect_extraction.ipynb` — Update Sections 1-2

**Current notebook Sections 1-2:**
- Cell 4 imports: `requests`, `BeautifulSoup`
- Loading: waits for `all_reviews.csv` from scraper

**Updated Sections 1-2:**
- Cell 4 imports: add `curl_cffi`
- New cell: `pip install curl_cffi` in Colab setup
- Load product list from embedded JSON (`products.json`)
- Inline scraping: use `curl_cffi.Session(impersonate='chrome124')` to fetch reviews from hidden API
- Save to `all_reviews.csv` + `products.json`
- Then proceed with existing preprocessing cells

### 5. `data_asg/bestbuy/products.json` — New file (curated product list)

Generated once, committed to repo. Contains ~35-40 Bluetooth speaker products with fields:

```json
[
  {
    "sku": "6509650",
    "name": "JBL Charge 5 - Portable Bluetooth Speaker",
    "price": 179.99,
    "rating": 4.7,
    "total_pages": 4447,
    "total_reviews": 88940
  },
  ...
]
```

---

## Implementation Steps

### Step 1: Generate curated product list

- From the 214 validated products, pick 40 unique models (one per model, not per color)
- Prioritize: well-known brands → high review count → diverse price range
- Order by review count so the scraper gets the best data first
- Save as `data_asg/bestbuy/products.json`

### Step 2: Rewrite `bestbuy_scraper.py`

- Remove: `requests`, `bs4`, `BeautifulSoup`, `parse_listing`, `parse_reviews`, `scrape_product_listing`
- Remove: `CATEGORY_URL`, `USER_AGENTS`
- Remove: `fetch_page()` (replaced by `fetch_reviews_api`)
- Add: `from curl_cffi import requests as curl_requests`
- Add: `_session = curl_requests.Session(impersonate="chrome124")`
- Add: `REVIEWS_API = "https://www.bestbuy.com/ugc/v2/reviews"`
- Add: `PRODUCT_LIST_PATH = "data_asg/bestbuy/products.json"`
- Add: `load_product_list()` function
- Add: `fetch_reviews_api()` function
- Add: `scrape_product_reviews()` function
- Keep: `save_product_data()`, `save_consolidated()`, `main()` — minimal changes
- Update `main()`: load product list → iterate → call `scrape_product_reviews()` → call `save_consolidated()`

### Step 3: Update tests

- `tests/conftest.py`: add `mock_api_response` fixture
- `tests/test_scraper.py`: replace BS4 tests with curl_cffi API tests
- Update `test_persistence.py`: keep as-is (placeholder)

### Step 4: Update notebook Sections 1-2

- Add `pip install curl_cffi` after existing pip installs
- Add import for curl_cffi
- Replace "load all_reviews.csv" with "scrape via API + save, then load"
- Ensure `df_valid` variable name matches downstream cells

### Step 5: Verify

- `python -c "import ast; ast.parse(open('bestbuy_scraper.py').read())"` — Syntax OK
- `python -c "from bestbuy_scraper import *; print('OK')"` — Module imports cleanly
- `pytest tests/test_scraper.py -v` — All tests pass
- Run scraper: `python bestbuy_scraper.py` — scrapes 1-2 products to confirm pipeline works end-to-end

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| curl_cffi imports fail (not installed) | Medium | High | Add to requirements.txt, Colab pip install |
| Hidden API changes schema | Low | High | Test `totalPages` + `topics[].text` field access first |
| Rate limiting after many requests | Medium | Medium | 2-3s delay between requests; resume with cached progress |
| Best Buy blocks the hidden API | Low | High | Fallback: bestbuy.ca with Selenium (teacher's approach) |
| Colab doesn't have curl_cffi | Medium | Medium | `pip install curl-cffi` — it's a pure Python package |

---

## Success Criteria

1. ✅ `bestbuy_scraper.py` imports and parses without errors
2. ✅ All tests pass (`pytest tests/ -v`)
3. ✅ `python bestbuy_scraper.py` scrapes at least 35 products with 100+ reviews each
4. ✅ Output `data_asg/bestbuy/all_reviews.csv` matches schema: product_name, review_text, rating, date
5. ✅ Output `data_asg/bestbuy/products.json` contains product metadata
6. ✅ Notebook loads scraped data and preprocessing cells execute without errors
7. ✅ At least 3500 reviews collected (35 products × 100 reviews)

---

## Estimated Effort

| Step | Time |
|------|------|
| Generate product list | 5 min (automated) |
| Rewrite scraper | 20 min |
| Update tests | 10 min |
| Update notebook | 15 min |
| Verify (dry run + tests) | 10 min |
| **Total** | **~60 min** |
