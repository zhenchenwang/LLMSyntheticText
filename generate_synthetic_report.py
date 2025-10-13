import argparse
import os
import sys
import requests
import shutil

# This script will be used to generate synthetic medical device reports.
# It will use the Hugging Face Inference API to interact with a Llama model.

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
DEMO_DIR = "demo"

def ensure_demo_dir_exists():
    """
    Ensures that the 'demo' directory exists, creating it if necessary.
    """
    try:
        os.makedirs(DEMO_DIR, exist_ok=True)
        print(f"Directory '{DEMO_DIR}' is ready.")
    except OSError as e:
        print(f"Error: Could not create directory '{DEMO_DIR}'. Reason: {e}")
        sys.exit(1)

def get_hf_api_token():
    """
    Retrieves the Hugging Face API token from the environment variables.

    Returns:
        str: The Hugging Face API token.

    Raises:
        SystemExit: If the HUGGING_FACE_HUB_TOKEN environment variable is not set.
    """
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: The HUGGING_FACE_HUB_TOKEN environment variable is not set.")
        print("Please set it to your Hugging Face API token.")
        sys.exit(1)
    return token

def query_model(prompt, api_token):
    """
    Sends a prompt to the Hugging Face Inference API and gets a response.

    Args:
        prompt (str): The prompt to send to the model.
        api_token (str): The Hugging Face API token.

    Returns:
        str: The generated text from the model, or None on failure.
    """
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "return_full_text": False,
        },
    }

    print("Sending request to Hugging Face API...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed: {e}")
        return None

    result = response.json()
    if isinstance(result, list) and result and "generated_text" in result[0]:
        return result[0]["generated_text"]
    else:
        print(f"Error: Unexpected API response format: {result}")
        return None

def read_file_content(filepath):
    """
    Reads the content of a text file.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The content of the file.

    Raises:
        SystemExit: If the file cannot be read.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def construct_output_path(copied_input_path):
    """
    Constructs the output file path for the synthetic report inside the demo directory.
    e.g., 'demo/report_1.txt' -> 'demo/synthetic_report_1.txt'

    Args:
        copied_input_path (str): The path to the copied input file in the demo directory.

    Returns:
        str: The constructed path for the output file.
    """
    base_name = os.path.basename(copied_input_path)
    if '.' in base_name:
        file_name, file_ext = base_name.rsplit('.', 1)
        output_filename = f"synthetic_{file_name}.{file_ext}"
    else:
        output_filename = f"synthetic_{base_name}"
    return os.path.join(DEMO_DIR, output_filename)

def save_report(content, output_path):
    """
    Saves the report content to the specified file path.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\nSuccessfully saved synthetic report to '{output_path}'")
    except IOError as e:
        print(f"\nError: Failed to save report to '{output_path}'. Reason: {e}")

def copy_input_to_demo(input_path):
    """
    Copies the input file to the 'demo' directory.

    Args:
        input_path (str): The path to the input file.

    Returns:
        str: The new path of the copied file in the demo directory.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        sys.exit(1)

    base_name = os.path.basename(input_path)
    new_path = os.path.join(DEMO_DIR, base_name)

    try:
        shutil.copy2(input_path, new_path)
        print(f"Copied input file to '{new_path}'")
        return new_path
    except shutil.Error as e:
        print(f"Error: Could not copy file to demo directory. Reason: {e}")
        sys.exit(1)

def main():
    """
    Main function to parse arguments and generate the report.
    """
    ensure_demo_dir_exists()
    api_token = get_hf_api_token()
    parser = argparse.ArgumentParser(
        description="Generate a synthetic medical device report using a real report as input."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="The path to the text file containing the real medical report.",
    )
    args = parser.parse_args()

    # 1. Copy the input file to the demo directory
    copied_input_path = copy_input_to_demo(args.input_file)

    # 2. Read the real report content from its new location
    print(f"Reading content from '{copied_input_path}'...")
    real_report_text = read_file_content(copied_input_path)

    # 2. Construct the prompt
    prompt = f"""Based on this real adverse event report, create a new, synthetic report with similar characteristics but different patient and event details.

Real Report:
---
{real_report_text}
---

Synthetic Report:
---
"""

    # 3. Query the model
    synthetic_report = query_model(prompt, api_token)

    # 4. Save the output
    if synthetic_report:
        clean_report = synthetic_report.strip()
        print("\n--- Generated Synthetic Report ---")
        print(clean_report)
        print("---------------------------------")

        output_path = construct_output_path(copied_input_path)
        save_report(clean_report, output_path)


if __name__ == "__main__":
    main()