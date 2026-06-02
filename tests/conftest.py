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
    """Return a sample product dict matching the scraper's output format."""
    return {
        "sku": "1234567",
        "name": "Test Bluetooth Speaker",
        "review_count": 2,
        "reviews": [
            {
                "title": "Great speaker",
                "text": "Really good sound for the price.",
                "rating": 5,
                "date": "2025-01-15",
                "author": "TestUser1",
            },
            {
                "title": "OK",
                "text": "Battery life could be better.",
                "rating": 3,
                "date": "2025-02-01",
                "author": "TestUser2",
            },
        ],
    }


@pytest.fixture
def sample_api_response():
    """Return a mock Best Buy hidden API response dict."""
    return {
        "totalPages": 3,
        "totalTopics": 45,
        "topics": [
            {
                "id": "abc-123",
                "topicType": "review",
                "rating": 5,
                "recommended": True,
                "title": "Amazing speaker",
                "text": "This speaker has incredible sound quality and bass.",
                "author": "HappyCustomer",
                "positiveFeedbackCount": 3,
                "negativeFeedbackCount": 0,
                "submissionTime": "2025-03-15T10:30:00.000-04:00",
            },
            {
                "id": "def-456",
                "topicType": "review",
                "rating": 2,
                "recommended": False,
                "title": "Disappointed",
                "text": "Stopped working after two weeks. Very poor build quality.",
                "author": "UnhappyBuyer",
                "positiveFeedbackCount": 0,
                "negativeFeedbackCount": 4,
                "submissionTime": "2025-03-10T14:15:00.000-04:00",
            },
            {
                "id": "ghi-789",
                "topicType": "question",  # non-review topic type
                "text": "Is this waterproof?",
                "author": "CuriousShopper",
                "submissionTime": "2025-03-08T09:00:00.000-04:00",
            },
        ],
    }
