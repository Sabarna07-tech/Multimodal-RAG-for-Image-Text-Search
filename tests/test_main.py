import os
import json
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# --- Test Setup ---

# Define test users and their API keys for a predictable testing environment.
TEST_API_KEYS = {
    "test-key-user-1": "test-user-1",
    "test-key-user-2": "test-user-2"
}

# Set environment variables for the test session.
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


def test_api_key_authentication(mocker):
    """
    Tests that the API key authentication is working correctly.
    """
    # Mock the external call to the Gemini API
    mock_response = MagicMock()
    mock_response.text = "This is a mocked response."
    mocker.patch("app.main.generator.model.generate_content", return_value=mock_response)

    chat_payload = {"thread_id": "test-thread", "message": "hello"}

    # Test with a valid API key
    response_valid = client.post(
        "/chat/",
        headers={"X-API-Key": "test-key-user-1"},
        json=chat_payload
    )
    assert response_valid.status_code == 200
    assert response_valid.json() == {"response": "This is a mocked response."}

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


def test_multi_tenancy_data_isolation_placeholder(mocker):
    """
    Tests that different users can access the chat endpoint, with the
    external API mocked.
    """
    # Mock the external call to the Gemini API
    mock_response = MagicMock()
    mock_response.text = "This is another mocked response."
    mocker.patch("app.main.generator.model.generate_content", return_value=mock_response)

    user1_key = "test-key-user-1"
    user2_key = "test-key-user-2"

    # Confirm both users can access the chat endpoint and get a valid (mocked) response.
    chat_payload_1 = {"thread_id": "user1-thread", "message": "test 1"}
    response_user1 = client.post("/chat/", headers={"X-API-Key": user1_key}, json=chat_payload_1)
    assert response_user1.status_code == 200
    assert response_user1.json() == {"response": "This is another mocked response."}

    chat_payload_2 = {"thread_id": "user2-thread", "message": "test 2"}
    response_user2 = client.post("/chat/", headers={"X-API-Key": user2_key}, json=chat_payload_2)
    assert response_user2.status_code == 200
    assert response_user2.json() == {"response": "This is another mocked response."}
