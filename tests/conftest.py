import pytest


@pytest.fixture
def sample_reviews():
    """Return sample speaker review strings for testing."""
    return [
        "Great sound quality and battery life",
        "Poor build quality, stops charging after 3 months",
        "Excellent value for money, clear audio",
    ]


@pytest.fixture
def tokenized_reviews(sample_reviews):
    """Return tokenized (split + lowercased) sample reviews."""
    return [r.split() for r in sample_reviews]


@pytest.fixture
def sample_product():
    """Return a sample product dict for output format testing."""
    return {
        "sku": "1234567",
        "name": "Test Bluetooth Speaker",
        "price": 49.99,
        "rating": 4.2,
        "review_count": 5,
        "reviews": [
            {
                "title": "Great speaker",
                "text": "Really good sound for the price.",
                "rating": 5,
                "date": "2025-01-15",
            },
            {
                "title": "OK",
                "text": "Battery life could be better.",
                "rating": 3,
                "date": "2025-02-01",
            },
        ],
    }
