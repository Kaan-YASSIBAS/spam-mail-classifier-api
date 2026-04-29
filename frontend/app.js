const form = document.getElementById("classificationForm");
const messageInput = document.getElementById("message");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");

const API_URL = "http://127.0.0.1:8000/predict";

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    resultBox.className = "result hidden";
    errorBox.classList.add("hidden");

    const message = messageInput.value.trim();

    if (!message) {
        errorBox.textContent = "Message cannot be empty.";
        errorBox.classList.remove("hidden");
        return;
    }

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            throw new Error("Classification request failed.");
        }

        const data = await response.json();
        const prediction = data.prediction;

        resultBox.textContent = `Prediction: ${prediction.toUpperCase()}`;
        resultBox.classList.remove("hidden");

        if (prediction === "spam") {
            resultBox.classList.add("spam");
        } else {
            resultBox.classList.add("ham");
        }

    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
    }
});