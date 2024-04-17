from flask import Flask, request, jsonify, render_template
import requests
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Model endpoints mapping
model_endpoints = {
    "fb-bart": "facebook/bart-large-cnn",
    "mistralai": "mistralai/Mistral-7B-Instruct-v0.2",
    "roberta": "deepset/roberta-base-squad2",
    "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
}

# Hugging Face API endpoint
API_URL = "https://api-inference.huggingface.co/models/"

def query_hf_model(model_name, payload):
    """
    Makes a POST request to the Hugging Face Inference API for a specified model.
    
    Parameters:
        model_name (str): The name of the model in the Hugging Face repository.
        payload (dict): The data payload to send in the POST request.
        
    Returns:
        dict: The JSON response from the API.
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"
    }
    try:
        response = requests.post(API_URL + model_name, headers=headers, json=payload)
        print(response.text)
        response.raise_for_status()  # Raises HTTPError for bad requests (4XX or 5XX)
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {"error": "Failed to query the model"}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate_answer():
    """
    Endpoint to generate answers from models based on the provided inputs.
    
    Expected JSON payload:
    {
        "model_key": "key identifying the model",
        "question": "text string or structured input depending on the model",
        "parameters": {}  # Optional parameters for certain models
    }
    
    Returns:
        JSON response with the model's answer or an error message.
    """
    data = request.json
    model_key = data.get("model_key")
    question = data.get("question")

    if not model_key or not question:
        return jsonify({"error": "Missing model_key or question"}), 400

    model_name = model_endpoints.get(model_key)
    if not model_name:
        return jsonify({"error": "Invalid model_key"}), 404

    if model_key == "roberta":
        payload = {"inputs": {"question": "Who was the person?", "context": str(question),}}
    else:
        payload = {"inputs": question}

    result = query_hf_model(model_name, payload)
    return jsonify(result)

if __name__ == '__main__':
    # Set the port number to 5000 or define using the 'PORT' environment variable
    port = int(os.getenv('PORT', 5000))
    # Run the Flask app without debug in production environments
    app.run(host='0.0.0.0', port=port, debug=True)
