import os
import json
from fastapi.testclient import TestClient

# --- Test Setup ---

# Define test users and their API keys for a predictable testing environment.
TEST_API_KEYS = {
    "test-key-user-1": "test-user-1",
    "test-key-user-2": "test-user-2"
}

# Set environment variables for the test session.
# This MUST be done BEFORE the application is imported to ensure Pydantic
# loads the correct settings for testing.
os.environ["GOOGLE_API_KEY"] = "dummy-test-api-key"
os.environ["API_KEYS"] = json.dumps(TEST_API_KEYS)

# Now that the environment is configured for testing, import the app.
from app.main import app

# Create a single TestClient instance to be used by all tests.
client = TestClient(app)


# --- Test Cases ---

def test_read_main_root():
    """
    Tests that the root endpoint is accessible.
    """
    response = client.get("/")
    assert response.status_code == 200


def test_api_key_authentication():
    """
    Tests that the API key authentication middleware is working correctly.
    - A valid key should be accepted.
    - An invalid key should be rejected with a 403 Forbidden error.
    - A missing key should be rejected with a 403 Forbidden error.
    """
    chat_payload = {"thread_id": "test-thread", "message": "hello"}

    # Test with a valid API key from our test set
    response_valid = client.post(
        "/chat/",
        headers={"X-API-Key": "test-key-user-1"},
        json=chat_payload
    )
    assert response_valid.status_code == 200

    # Test with an invalid API key
    response_invalid = client.post(
        "/chat/",
        headers={"X-API-Key": "this-key-is-not-valid"},
        json=chat_payload
    )
    assert response_invalid.status_code == 403
    assert response_invalid.json() == {"detail": "Invalid API Key"}

    # Test with no API key provided
    response_no_key = client.post("/chat/", json=chat_payload)
    assert response_no_key.status_code == 403


def test_multi_tenancy_data_isolation_placeholder():
    """
    This test serves as a structural placeholder for verifying multi-tenancy.

    A complete version of this test would be more complex, requiring mocking of
    the YouTube/PDF data extractors and the AI generation model to avoid
    real network calls.

    The test would:
    1. Mock `extract_youtube_data`.
    2. Have User 1 ingest data about "Topic A".
    3. Have User 2 ingest data about "Topic B".
    4. Verify (by checking the context passed to the generator) that a chat
       from User 1 only uses "Topic A" data and a chat from User 2 only
       uses "Topic B" data.

    For now, this simplified test confirms that the endpoints can be reached
    by different authenticated users, which is a prerequisite for multi-tenancy.
    """
    user1_key = "test-key-user-1"
    user2_key = "test-key-user-2"

    # Confirm both users can access the chat endpoint with their respective keys.
    # This implies the backend is correctly identifying them.
    chat_payload_1 = {"thread_id": "user1-thread", "message": "test 1"}
    response_user1 = client.post("/chat/", headers={"X-API-Key": user1_key}, json=chat_payload_1)
    assert response_user1.status_code == 200

    chat_payload_2 = {"thread_id": "user2-thread", "message": "test 2"}
    response_user2 = client.post("/chat/", headers={"X-API-Key": user2_key}, json=chat_payload_2)
    assert response_user2.status_code == 200
