import bs4  # BeautifulSoup for parsing HTML
import os
import json
import time
import datetime 
import sys
from app import query_hf_model

def read_and_process_html(html_path):
    """
    Reads an HTML file and extracts its text content, ignoring HTML tags.

    Parameters:
        html_path (str): The file path to the HTML document.

    Returns:
        str: Formatted text extracted from the HTML.
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = bs4.BeautifulSoup(file, 'html.parser')
            text = soup.get_text()
            processed_text = " ".join(text.split()).replace("\n", " ").strip()
        return processed_text
    except Exception as e:
        print(f"Failed to process HTML file: {e}")
        return None

# Model endpoints mapping
model_endpoints = {
    "fb-bart": "facebook/bart-large-cnn",
    "falcon": "Falconsai/text_summarization",
    "mistralai": "mistralai/Mistral-7B-Instruct-v0.2",
    "roberta": "deepset/roberta-base-squad2",
    # "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
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

def query_models_with_html_input(html_path):
    """
    Extracts text from an HTML file, formats it, and queries all models with this text.

    Parameters:
        html_path (str): The file path to the HTML document.
    
    Returns:
        dict: A dictionary with model keys and their API responses.
    """
    processed_text = read_and_process_html(html_path)
    if processed_text is None:
        return {"error": "Could not process the HTML file."}

    custom_queries = {
        "fb-bart": {"inputs": processed_text},
        "falcon": {"inputs": processed_text},
        "mistralai": {"inputs": processed_text},
        "roberta": {"inputs": {"question": "Who is mentioned in the document?", "context": processed_text}},
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


def save_results_to_html(results, output_file):
    """
    Saves the given results dictionary to an HTML file formatted with Bootstrap.

    Parameters:
        results (dict): The results to save.
        output_file (str): The path to the output HTML file.
    """
    try:
        with open(output_file, 'w') as file:
            file.write("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Query Results</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
<div class="container mt-5 ml-5 mr-5 mb-5">
    <h1 class="mb-4" style="font-weight: 800;">Model Query Results</h1>
""")
            for model_key, data in results.items():
                file.write(f'<h2 style="margin: 50px 5px 5px 10px; font-weight: 600;">{model_key.replace("-", " ").title()}</h2>\n')
                if isinstance(data, dict) and "error" in data:
                    file.write(f'<div class="alert alert-danger" role="alert">{data["error"]}</div>\n')
                elif isinstance(data, list):
                    for item in data:
                        for key, val in item.items():
                            file.write(f'<div class="card mb-3"><div class="card-body"><h5 class="card-title">{key.replace("_", " ").title()}</h5><p class="card-text">{val}</p></div></div>\n')
                elif isinstance(data, dict):
                    file.write('<ul class="list-group mb-3">\n')
                    for key, val in data.items():
                        file.write(f'<li class="list-group-item"><strong>{key.title()}:</strong> {val}</li>\n')
                    file.write('</ul>\n')
            file.write("""
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
""")
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Failed to save results to HTML: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 app.py <input.html> <results.html>")
        sys.exit(1)
    html_path = sys.argv[1]
    output_path = sys.argv[2]

    results = query_models_with_html_input(html_path)
    save_results_to_html(results, output_path)
