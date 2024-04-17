import fitz  # PyMuPDF
import os
import json
import time
from app import query_hf_model

def read_and_process_pdf(pdf_path):
    """
    Reads a PDF file and extracts its text content in a formatted string.

    Parameters:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: Formatted text extracted from the PDF.
    """
    try:
        document = fitz.open(pdf_path)
        text = []
        for page in document:
            text.append(page.get_text())
        document.close()
        processed_text = " ".join(text).replace("\n", " ").strip()
        return processed_text
    except Exception as e:
        print(f"Failed to process PDF file: {e}")
        return None

# Model endpoints mapping
model_endpoints = {
    "fb-bart": "facebook/bart-large-cnn",
    "falcon": "Falconsai/text_summarization",
    "mistralai": "mistralai/Mistral-7B-Instruct-v0.2",
    "roberta": "deepset/roberta-base-squad2",
    "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
    "pegasus": "starcatmeow/autotrain-cybersecurity-summarization-pegasus-x-book-43369110299",
}

def query_hf_model_with_retries(model_name, payload, max_retries=3):
    """
    Queries the specified model and handles retries if the response status is not 200.

    Parameters:
        model_name (str): The model endpoint name.
        payload (dict): The payload to send in the query.
        max_retries (int): Maximum number of retries allowed.

    Returns:
        dict or list: The response from the model query.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"Attempting to query {model_name}, attempt {attempt + 1}")
            response = query_hf_model(model_name, payload)
            if isinstance(response, dict) and 'error' in response:
                raise Exception(f"Received error response: {response['error']}")
            # Check if there's a 'status' key explicitly indicating a problem
            if isinstance(response, dict) and response.get('status') not in (200, None):
                raise Exception(f"Received non-200 status: {response.get('status')}")
            return response
        except Exception as e:
            print(f"Error querying {model_name}: {e}")
            attempt += 1
            if attempt < max_retries:
                time.sleep(10)  # Wait for 10 seconds before next attempt
            else:
                print(f"Max retries reached for {model_name}.")
                return {"error": f"Max retries reached with error: {e}"}

def query_all_models(example_queries):
    results = {}
    for model_key, model_name in model_endpoints.items():
        payload = example_queries.get(model_key)
        if payload:
            response = query_hf_model_with_retries(model_name, payload)
            results[model_key] = response
            print(f"Response from {model_key}: {response}\n")
    return results

def query_models_with_pdf_input(pdf_path):
    """
    Extracts text from a PDF file, formats it, and queries all models with this text.

    Parameters:
        pdf_path (str): The file path to the PDF document.
    
    Returns:
        dict: A dictionary with model keys and their API responses.
    """
    processed_text = read_and_process_pdf(pdf_path)
    if processed_text is None:
        return {"error": "Could not process the PDF file."}

    custom_queries = {
        "fb-bart": {"inputs": processed_text},
        "falcon": {"inputs": processed_text},
        "mistralai": {"inputs": processed_text},
        "roberta": {"inputs": {"question": "Who is mentioned in the document?", "context": processed_text}},
        "distilbert-sst2": {"inputs": processed_text},
        "pegasus": {"inputs": processed_text},
    }
    results = query_all_models(custom_queries)
    return results

def save_results_to_json(results, output_file):
    """
    Saves the given results dictionary to a JSON file.

    Parameters:
        results (dict or list): The results to save.
        output_file (str): The path to the output JSON file.
    """
    try:
        with open(output_file, 'w') as file:
            json.dump(results, file, indent=4)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Failed to save results to JSON: {e}")

if __name__ == "__main__":
    pdf_path = 'sample.pdf'
    output_path = 'results.json'

    results = query_models_with_pdf_input(pdf_path)
    save_results_to_json(results, output_path)
