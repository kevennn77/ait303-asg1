"""Tests for save/load persistence functions in the scraper.

These tests validate that save_product_data and save_consolidated produce
correct file output without requiring live API calls.
"""

import json
import csv
import tempfile
import os
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Test: save_product_data
# ---------------------------------------------------------------------------


def test_save_product_data_creates_json():
    """save_product_data should write a JSON file with expected keys."""
    from bestbuy_scraper import save_product_data

    reviews = [
        {"title": "R1", "text": "Review one", "rating": 4,
         "date": "2025-01-01", "author": "A"},
        {"title": "R2", "text": "Review two", "rating": 5,
         "date": "2025-02-01", "author": "B"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_product_data("99999", "Test Product", reviews, output_dir=tmpdir)

        path = os.path.join(tmpdir, "99999.json")
        assert os.path.exists(path), f"Expected {path} to exist"

        with open(path) as f:
            data = json.load(f)

    assert data["sku"] == "99999"
    assert data["name"] == "Test Product"
    assert len(data["reviews"]) == 2
    assert data["reviews"][0]["text"] == "Review one"
    assert data["reviews"][1]["rating"] == 5


def test_save_product_data_handles_empty_reviews():
    """save_product_data should handle products with zero reviews."""
    from bestbuy_scraper import save_product_data

    with tempfile.TemporaryDirectory() as tmpdir:
        save_product_data("00000", "No Reviews", [], output_dir=tmpdir)
        path = os.path.join(tmpdir, "00000.json")
        with open(path) as f:
            data = json.load(f)
    assert data["sku"] == "00000"
    assert data["reviews"] == []


def test_save_product_data_creates_directory():
    """save_product_data should create the output directory if missing."""
    from bestbuy_scraper import save_product_data

    with tempfile.TemporaryDirectory() as tmpdir:
        nested = os.path.join(tmpdir, "new", "nested", "dir")
        save_product_data("111", "Test", [], output_dir=nested)
        path = os.path.join(nested, "111.json")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Test: save_consolidated
# ---------------------------------------------------------------------------


def test_save_consolidated_creates_csv():
    """save_consolidated should write a CSV with correct headers."""
    from bestbuy_scraper import save_consolidated

    products = [{"sku": "123", "name": "Prod A", "review_count": 2}]
    reviews = [
        {"product_name": "Prod A", "review_text": "Good", "rating": 5, "date": "2025-01-01"},
        {"product_name": "Prod A", "review_text": "Bad", "rating": 1, "date": "2025-02-01"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_consolidated(products, reviews, output_dir=tmpdir)

        # Check CSV
        csv_path = os.path.join(tmpdir, "all_reviews.csv")
        assert os.path.exists(csv_path)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert reader.fieldnames == ["product_name", "review_text", "rating", "date"]
        assert len(rows) == 2
        assert rows[0]["review_text"] == "Good"
        assert rows[1]["rating"] == "1"

        # Check products catalog JSON
        cat_path = os.path.join(tmpdir, "products.json")
        assert os.path.exists(cat_path)
        with open(cat_path) as f:
            catalog = json.load(f)
        assert len(catalog) == 1
        assert catalog[0]["sku"] == "123"
        assert catalog[0]["review_count"] == 2


def test_save_consolidated_empty():
    """save_consolidated should handle empty review lists."""
    from bestbuy_scraper import save_consolidated

    with tempfile.TemporaryDirectory() as tmpdir:
        save_consolidated([], [], output_dir=tmpdir)

        csv_path = os.path.join(tmpdir, "all_reviews.csv")
        assert os.path.exists(csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows == []


# ---------------------------------------------------------------------------
# Test: main entry points (dry-run simulated via CLI args)
# ---------------------------------------------------------------------------


@patch("bestbuy_scraper.load_product_list")
@patch("bestbuy_scraper.fetch_reviews_page")
@patch("bestbuy_scraper.save_consolidated")
def test_main_dry_run_no_file_write(mock_save, mock_fetch, mock_load):
    """main() with --dry-run should not write files."""
    from bestbuy_scraper import main

    mock_load.return_value = [
        {"sku": "111", "name": "Product 1"},
    ]
    mock_fetch.return_value = {
        "totalPages": 1,
        "topics": [
            {
                "id": "t1",
                "topicType": "review",
                "title": "Test",
                "text": "Review text",
                "rating": 4,
                "submissionTime": "2025-01-01T00:00:00.000Z",
                "author": "Tester",
            }
        ],
    }

    with patch.object(sys := __import__("sys"), "argv", ["bestbuy_scraper.py", "--dry-run"]):
        main()

    # save_consolidated should NOT be called in dry-run mode
    mock_save.assert_not_called()
