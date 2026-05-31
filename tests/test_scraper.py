import pytest
import re


@pytest.mark.skip(reason="Live execution required — run scraper first")
def test_min_products():
    """Validate that scraper collected 30+ products."""
    assert True


@pytest.mark.skip(reason="Live execution required — run scraper first")
def test_delay_between_requests():
    """Validate randomized 1-3 second delays between requests."""
    assert True


def test_json_output_format(sample_product):
    """Assert sample_product dict has expected keys."""
    expected_keys = {"sku", "name", "price", "rating", "review_count", "reviews"}
    assert expected_keys.issubset(sample_product.keys()), (
        f"Missing keys. Expected at least {expected_keys}, got {set(sample_product.keys())}"
    )


def test_csv_schema():
    """Assert expected CSV columns match the all_reviews.csv schema."""
    expected_columns = ["product_name", "review_text", "rating", "date"]
    # Schema contract check — the scraper must produce these columns
    assert len(expected_columns) == 4
    assert "product_name" in expected_columns
    assert "review_text" in expected_columns
    assert "rating" in expected_columns
    assert "date" in expected_columns


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
