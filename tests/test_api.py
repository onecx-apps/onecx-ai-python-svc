from fastapi.testclient import TestClient
from agent.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

def test_start_conversation():
    data = {"conversation_type": "Q_AND_A"}
    response = client.post("/startConversation", json=data)
    assert response.status_code == 200
    assert "conversationId" in response.json()
    assert "history" in response.json()
    assert "conversationType" in response.json()

def test_flow():
    data = {"conversation_type": "Q_AND_A"}
    response = client.post("/startConversation", json=data)

    conversationId = response.json()["conversationId"]
    print(conversationId)

    second_response = client.get(f"/conversation/{conversationId}")
    assert second_response.status_code == 200
    assert "conversationId" in second_response.json()
    assert "history" in second_response.json()
    assert "conversationType" in second_response.json()

