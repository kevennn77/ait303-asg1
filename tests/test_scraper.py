"""Unit tests for the curl_cffi-based Best Buy scraper.

Tests the internal functions that can be tested without live API calls:
  - _parse_topic
  - fetch_reviews_page (via mock)
  - scrape_product_reviews (via mock)
  - Output format contracts (CSV schema, product catalog)
"""

from unittest.mock import patch, Mock
import json
import pytest
import re

# ---------------------------------------------------------------------------
# Test: _parse_topic
# ---------------------------------------------------------------------------


def test_parse_topic_extracts_review_fields(sample_api_response):
    """_parse_topic should extract all expected fields from a review topic."""
    from bestbuy_scraper import _parse_topic

    topic = sample_api_response["topics"][0]
    result = _parse_topic(topic)

    assert result["title"] == "Amazing speaker"
    assert result["text"] == "This speaker has incredible sound quality and bass."
    assert result["rating"] == 5
    assert result["date"] == "2025-03-15T10:30:00.000-04:00"
    assert result["author"] == "HappyCustomer"


def test_parse_topic_handles_missing_fields():
    """_parse_topic should gracefully handle missing optional fields."""
    from bestbuy_scraper import _parse_topic

    topic = {"topicType": "review"}  # minimal topic
    result = _parse_topic(topic)

    assert result["title"] == ""
    assert result["text"] == ""
    assert result["rating"] is None
    assert result["date"] == ""
    assert result["author"] == ""


def test_parse_topic_skips_questions(sample_api_response):
    """_parse_topic should still parse a question topic but return its fields."""
    from bestbuy_scraper import _parse_topic

    topic = sample_api_response["topics"][2]  # topicType == "question"
    result = _parse_topic(topic)

    # This function doesn't filter by topicType — filtering happens in scrape_product_reviews
    assert result["text"] == "Is this waterproof?"


# ---------------------------------------------------------------------------
# Test: fetch_reviews_page
# ---------------------------------------------------------------------------


@patch("bestbuy_scraper._session.get")
def test_fetch_reviews_page_success(mock_get, sample_api_response):
    """fetch_reviews_page should return parsed JSON on 200."""
    from bestbuy_scraper import fetch_reviews_page

    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sample_api_response
    mock_get.return_value = mock_resp

    result = fetch_reviews_page("1234567", 1)
    assert result == sample_api_response


@patch("bestbuy_scraper._session.get")
def test_fetch_reviews_page_error_status(mock_get):
    """fetch_reviews_page should return None on non-200 status."""
    from bestbuy_scraper import fetch_reviews_page

    mock_resp = Mock()
    mock_resp.status_code = 403
    mock_get.return_value = mock_resp

    result = fetch_reviews_page("1234567", 1)
    assert result is None


@patch("bestbuy_scraper._session.get")
def test_fetch_reviews_page_exception(mock_get):
    """fetch_reviews_page should return None on network error."""
    from bestbuy_scraper import fetch_reviews_page

    mock_get.side_effect = Exception("Connection refused")

    result = fetch_reviews_page("1234567", 1)
    assert result is None


# ---------------------------------------------------------------------------
# Test: scrape_product_reviews
# ---------------------------------------------------------------------------


@patch("bestbuy_scraper.fetch_reviews_page")
def test_scrape_product_reviews_collects(mock_fetch, sample_api_response):
    """scrape_product_reviews should collect reviews across pages."""
    from bestbuy_scraper import scrape_product_reviews

    # First page returns 2 reviews + totalPages=3
    page1 = dict(sample_api_response)
    page1["totalPages"] = 3
    # Second page returns 1 more review
    page2 = {
        "totalPages": 3,
        "topics": [
            {
                "id": "jkl-012",
                "topicType": "review",
                "rating": 4,
                "title": "Pretty good",
                "text": "Decent speaker for the price.",
                "author": "SatisfiedUser",
                "submissionTime": "2025-03-01T08:00:00.000-04:00",
            }
        ],
    }
    # Third page returns empty
    page3 = {"totalPages": 3, "topics": []}

    mock_fetch.side_effect = [page1, page2, page3]

    reviews = scrape_product_reviews("1234567", max_reviews=10)
    assert len(reviews) == 3  # 2 from page 1 + 1 from page 2
    assert reviews[0]["title"] == "Amazing speaker"
    assert reviews[1]["title"] == "Disappointed"
    assert reviews[2]["title"] == "Pretty good"


@patch("bestbuy_scraper.fetch_reviews_page")
def test_scrape_product_reviews_excludes_questions(mock_fetch, sample_api_response):
    """scrape_product_reviews should only include topicType == 'review'."""
    from bestbuy_scraper import scrape_product_reviews

    # Single-page response so loop doesn't repeat
    single_page = dict(sample_api_response)
    single_page["totalPages"] = 1
    mock_fetch.return_value = single_page  # includes 2 reviews + 1 question
    reviews = scrape_product_reviews("1234567", max_reviews=10)
    assert len(reviews) == 2
    assert all(r["title"] in ("Amazing speaker", "Disappointed") for r in reviews)


@patch("bestbuy_scraper.fetch_reviews_page")
def test_scrape_product_reviews_respects_max(mock_fetch, sample_api_response):
    """scrape_product_reviews should stop at max_reviews."""
    from bestbuy_scraper import scrape_product_reviews

    mock_fetch.return_value = sample_api_response  # 2 reviews per page
    reviews = scrape_product_reviews("1234567", max_reviews=1)
    assert len(reviews) == 1


@patch("bestbuy_scraper.fetch_reviews_page")
def test_scrape_product_reviews_stops_on_empty(mock_fetch, sample_api_response):
    """scrape_product_reviews should stop when API returns empty topics."""
    from bestbuy_scraper import scrape_product_reviews

    page1 = dict(sample_api_response)
    page1["totalPages"] = 1
    mock_fetch.return_value = page1
    reviews = scrape_product_reviews("1234567", max_reviews=100)
    assert len(reviews) == 2


# ---------------------------------------------------------------------------
# Test: Output format contracts
# ---------------------------------------------------------------------------


def test_csv_schema():
    """Assert expected CSV columns match the all_reviews.csv schema."""
    from bestbuy_scraper import save_consolidated

    expected_columns = ["product_name", "review_text", "rating", "date"]
    # This is a schema contract — verify the save function uses these column names
    # We inspect the function's source to confirm fieldnames
    import inspect

    source = inspect.getsource(save_consolidated)
    for col in expected_columns:
        assert col in source, f"Column '{col}' not found in save_consolidated source"


def test_product_json_schema():
    """Assert per-product JSON has the expected keys."""
    import tempfile, os, json

    from bestbuy_scraper import save_product_data

    reviews = [
        {
            "title": "Test",
            "text": "Review text",
            "rating": 5,
            "date": "2025-01-01",
            "author": "Tester",
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_product_data("99999", "Test Product", reviews, output_dir=tmpdir)
        path = os.path.join(tmpdir, "99999.json")
        with open(path) as f:
            data = json.load(f)

    assert data["sku"] == "99999"
    assert data["name"] == "Test Product"
    assert len(data["reviews"]) == 1
    assert data["reviews"][0]["text"] == "Review text"


def test_load_product_list(tmp_path):
    """load_product_list should read and validate the product catalog."""
    from bestbuy_scraper import load_product_list

    products = [
        {"sku": "111", "name": "Product 1"},
        {"sku": "222", "name": "Product 2"},
    ]
    path = tmp_path / "products.json"
    with open(path, "w") as f:
        json.dump(products, f)

    result = load_product_list(str(path))
    assert len(result) == 2
    assert result[0]["sku"] == "111"
    assert result[1]["name"] == "Product 2"


# ---------------------------------------------------------------------------
# Test: PII check (data quality)
# ---------------------------------------------------------------------------


def test_no_pii_in_scraped_data(sample_reviews):
    """Assert sample reviews contain no phone numbers or email addresses."""
    phone_pattern = re.compile(r"\d{3}[-.]?\d{3}[-.]?\d{4}")
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    for review in sample_reviews:
        assert phone_pattern.search(review) is None, (
            f"Phone number pattern found in: {review[:50]}"
        )
        assert email_pattern.search(review) is None, (
            f"Email pattern found in: {review[:50]}"
        )


# ---------------------------------------------------------------------------
# Test: API endpoint has required fields
# ---------------------------------------------------------------------------


def test_api_response_has_required_fields(sample_api_response):
    """Assert the API response shape matches expectations."""
    assert "totalPages" in sample_api_response
    assert "topics" in sample_api_response

    for topic in sample_api_response["topics"]:
        # Every topic should have an id
        assert "id" in topic
        # Review-type topics should have rating
        if topic.get("topicType") == "review":
            assert "rating" in topic
            assert "text" in topic
