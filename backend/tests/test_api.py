from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Spam Mail Classifier API is running"
    }


def test_predict_spam_message():
    payload = {
        "message": "Congratulations! You won a free prize. Click now!"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": "spam"
    }


def test_predict_ham_message():
    payload = {
        "message": "Can you send me the homework file?"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": "ham"
    }


def test_predict_empty_message():
    payload = {
        "message": ""
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_missing_message_field():
    payload = {}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422