# Synthetic Medical Data Generation and Evaluation Toolkit

## 1. Project Overview

This project provides a complete toolkit for generating synthetic medical device adverse event reports and evaluating their quality across three key pillars: **Privacy & Novelty**, **Fidelity & Utility**, and **Fluency & Quality**.

The primary goal is to create high-quality synthetic data that is statistically similar to real-world data but contains no personally identifiable information (PII), making it safe for research, development, and testing purposes.

This toolkit includes:
*   A script to generate synthetic reports using a large language model (LLM) via an API.
*   A comprehensive evaluation suite to measure the quality of the generated data.
*   Example data and a "Train on Synthetic, Test on Real" (TSTR) simulation to demonstrate the utility of the synthetic data.

---

## 2. Core Concepts

### What is Synthetic Data?
Synthetic data is artificially generated data that is not based on real-world events. In this project, we use a Large Language Model (LLM) to read real medical reports and generate new, similar reports that are entirely fictional. This allows us to create large, realistic datasets without compromising the privacy of real individuals.

### The Three Pillars of Synthetic Data Evaluation
A robust evaluation of synthetic data quality rests on three key pillars:

**Pillar 1: Privacy & Novelty**
This pillar answers the question: "Is the data safe and original?"
*   **Privacy:** We must ensure that no sensitive information (like names, locations, or specific dates) from the real data has leaked into the synthetic data. We use PII scanning tools to verify this.
*   **Novelty:** The synthetic data should be new and not just a copy or simple paraphrase of the original. We measure this using lexical (word-based) and semantic (meaning-based) similarity scores. Low scores are better.

**Pillar 2: Fidelity & Utility**
This is the most critical pillar. It answers: "Is the synthetic data actually useful?"
*   **Fidelity:** The synthetic data should have the same statistical characteristics as the real data. We compare properties like text length, vocabulary richness, and word frequency distributions.
*   **Utility:** The ultimate test is whether the synthetic data can be used for a real-world task. We use a **Train on Synthetic, Test on Real (TSTR)** evaluation. We train one machine learning model on real data and an identical model on synthetic data. We then compare their performance on a real test set. A small performance gap means the synthetic data has high utility.

**Pillar 3: Fluency & Quality**
This pillar answers: "Is the generated text readable and coherent?"
*   **Fluency:** The text should be grammatically correct and sound natural.
*   **Coherence:** The text should make logical sense.
*(Note: Pillar 3 evaluation is planned as a future extension for this project.)*

---

## 3. How to Use This Toolkit

### Step 1: Setup and Installation

First, clone the repository and install the necessary Python libraries.

```bash
# Install all required packages
pip install -r requirements.txt

# Download the English language model for spaCy (used by the PII scanner)
python -m spacy download en_core_web_lg
```

### Step 2: Set Your API Token

To generate new data, you need a Hugging Face API token. Set it as an environment variable in your terminal.

**For Linux/macOS:**
```bash
export HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"
```
**For Windows (Command Prompt):**
```bash
set HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"
```

### Step 3: Generate a New Synthetic Report

Use the `generate_synthetic_report.py` script to create a new synthetic report from an input file.

```bash
# Create a sample input file
echo "This is a test report about a device malfunction." > my_report.txt

# Run the generation script
python generate_synthetic_report.py my_report.txt
```
This will create a `demo` folder, copy your input file into it, and save the newly generated synthetic report inside as well.

### Step 4: Evaluate the Generated Report

Use the `evaluate_privacy.py` script to run the comprehensive Pillar 1 and Statistical Fidelity evaluation. Point it to the original and synthetic files inside the `demo` folder.

```bash
python evaluate_privacy.py demo/my_report.txt demo/synthetic_my_report.txt
```
The results will be printed to your console and also saved to a new file named `demo/evaluation_my_report.txt`.

### Step 5: Run the TSTR Utility Evaluation

Use the `evaluate_tstr.py` script to run the "Train on Synthetic, Test on Real" simulation. This script uses the pre-built datasets in the `tstr_data` directory.

```bash
python evaluate_tstr.py
```
This will train the two models, compare their performance, and save the final report to `demo/tstr_evaluation_report.txt`.

---

## 4. Understanding the Output

*   **`demo/`**: This directory contains all the generated output files for easy demonstration.
*   **`tstr_data/`**: This directory contains the pre-built, labeled datasets used for the TSTR simulation.
*   **Evaluation Reports (`.txt` files):** These files provide a detailed breakdown of all the privacy, novelty, and fidelity metrics, allowing you to assess the quality of your synthetic data at a glance. A good result will have **zero PII leakage**, **low similarity scores**, and **closely matched statistical distributions**. The TSTR report should show a **low performance gap**.