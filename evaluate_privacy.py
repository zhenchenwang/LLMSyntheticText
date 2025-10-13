import argparse
import sys
import os
from presidio_analyzer import AnalyzerEngine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

def read_file_content(filepath):
    """
    Reads and returns the content of a file. Exits if the file cannot be found.
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

def analyze_pii_leakage(original_text, synthetic_text):
    """
    Analyzes the synthetic text for any PII found in the original text.
    Returns the analysis results as a string.
    """
    report_lines = ["--- 1. PII Leakage Analysis ---"]
    try:
        analyzer = AnalyzerEngine()
        report_lines.append("Analyzing original text for PII...")
        original_pii_results = analyzer.analyze(text=original_text, language='en')
    except Exception as e:
        report_lines.append(f"\nError: Presidio analysis failed. Please ensure you have downloaded the spaCy model:")
        report_lines.append("python -m spacy download en_core_web_lg")
        report_lines.append(f"Underlying error: {e}")
        return "\n".join(report_lines)

    if not original_pii_results:
        report_lines.append("No PII entities found in the original document.")
        return "\n".join(report_lines)

    report_lines.append(f"Found {len(original_pii_results)} PII entities in the original document.")

    leaked_pii_count = 0
    leaked_items = []
    for pii_result in original_pii_results:
        pii_text = original_text[pii_result.start:pii_result.end]
        if pii_text in synthetic_text:
            leaked_items.append(f"  [!] Leaked PII found: '{pii_text}' (Type: {pii_result.entity_type})")
            leaked_pii_count += 1

    if leaked_pii_count == 0:
        report_lines.append("[+] No PII from the original document was found in the synthetic document.")
    else:
        report_lines.extend(leaked_items)
        report_lines.append(f"\nSummary: Found {leaked_pii_count} leaked PII entities.")

    report_lines.append("-" * 20)
    return "\n".join(report_lines)


def analyze_statistical_fidelity(original_text, synthetic_text):
    """
    Analyzes and compares the statistical properties of the two texts.
    """
    report_lines = ["\n--- 5. Statistical Fidelity Analysis ---"]

    try:
        # Download NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            report_lines.append("Downloading NLTK 'punkt' model...")
            nltk.download('punkt', quiet=True)

        # Tokenize texts
        original_words = word_tokenize(original_text.lower())
        synthetic_words = word_tokenize(synthetic_text.lower())
        original_sents = sent_tokenize(original_text)
        synthetic_sents = sent_tokenize(synthetic_text)

        # 1. Text Length Comparison
        report_lines.append("a) Text Length:")
        report_lines.append(f"  - Word Count: Original={len(original_words)} vs. Synthetic={len(synthetic_words)}")
        report_lines.append(f"  - Sentence Count: Original={len(original_sents)} vs. Synthetic={len(synthetic_sents)}")

        # 2. Vocabulary Richness (Type-Token Ratio)
        ttr_original = len(set(original_words)) / len(original_words) if len(original_words) > 0 else 0
        ttr_synthetic = len(set(synthetic_words)) / len(synthetic_words) if len(synthetic_words) > 0 else 0
        report_lines.append("\nb) Vocabulary Richness (Type-Token Ratio):")
        report_lines.append(f"  - Original: {ttr_original:.4f}")
        report_lines.append(f"  - Synthetic: {ttr_synthetic:.4f}")

        # 3. Word Frequency Distribution (Top 20 words)
        report_lines.append("\nc) Top 20 Word Frequency Comparison:")
        original_counts = Counter(original_words)
        synthetic_counts = Counter(synthetic_words)
        report_lines.append(f"  {'Word':<15} | {'Original Freq':<15} | {'Synthetic Freq':<15}")
        report_lines.append(f"  {'-'*15} | {'-'*15} | {'-'*15}")
        for word, count in original_counts.most_common(20):
            synth_freq = synthetic_counts.get(word, 0)
            report_lines.append(f"  {word:<15} | {count:<15} | {synth_freq:<15}")

    except Exception as e:
        report_lines.append(f"Error during statistical fidelity analysis: {e}")

    report_lines.append("-" * 20)
    return "\n".join(report_lines)


def analyze_text_similarity(original_text, synthetic_text):
    """
    Calculates the cosine similarity between the two texts and returns the report as a string.
    """
    report_lines = ["\n--- 2. Text Similarity Analysis ---"]
    try:
        vectorizer = TfidfVectorizer().fit_transform([original_text, synthetic_text])
        similarity_matrix = cosine_similarity(vectorizer)
        score = similarity_matrix[0, 1]

        report_lines.append(f"Cosine Similarity Score: {score:.4f}")

        if score > 0.7:
            report_lines.append("[!] High similarity. The synthetic report is very close to the original.")
        elif score > 0.4:
            report_lines.append("[~] Moderate similarity. The synthetic report shares some content with the original.")
        else:
            report_lines.append("[+] Low similarity. The synthetic report is significantly different from the original.")

    except Exception as e:
        report_lines.append(f"Error during text similarity analysis: {e}")

    report_lines.append("-" * 20)
    return "\n".join(report_lines)


def analyze_lexical_similarity(original_text, synthetic_text):
    """
    Calculates the ROUGE-L score and returns the report as a string.
    """
    report_lines = ["\n--- 3. Lexical Similarity (ROUGE-L) ---"]
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(original_text, synthetic_text)
        rouge_l_fmeasure = scores['rougeL'].fmeasure

        report_lines.append(f"ROUGE-L F-measure: {rouge_l_fmeasure:.4f}")

        if rouge_l_fmeasure > 0.6:
            report_lines.append("[!] High lexical similarity. The synthetic report may contain copied sentences.")
        elif rouge_l_fmeasure > 0.3:
            report_lines.append("[~] Moderate lexical similarity. Some phrases may be copied.")
        else:
            report_lines.append("[+] Low lexical similarity. The text is likely original.")

    except Exception as e:
        report_lines.append(f"Error during ROUGE score calculation: {e}")

    report_lines.append("-" * 20)
    return "\n".join(report_lines)


def analyze_semantic_similarity(original_text, synthetic_text):
    """
    Calculates semantic similarity using Sentence-BERT and returns the report as a string.
    """
    report_lines = ["\n--- 4. Semantic Similarity (Sentence-BERT) ---"]
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_1 = model.encode(original_text, convert_to_tensor=True)
        embedding_2 = model.encode(synthetic_text, convert_to_tensor=True)
        score = util.cos_sim(embedding_1, embedding_2).item()

        report_lines.append(f"Semantic Similarity Score: {score:.4f}")

        if score > 0.8:
            report_lines.append("[!] High semantic similarity. The synthetic report is very likely a paraphrase.")
        elif score > 0.5:
            report_lines.append("[~] Moderate semantic similarity. The reports share similar meaning.")
        else:
            report_lines.append("[+] Low semantic similarity. The reports are likely distinct in meaning.")

    except Exception as e:
        report_lines.append(f"Error during semantic similarity analysis: {e}")

    report_lines.append("-" * 20)
    return "\n".join(report_lines)


def save_evaluation_report(report_content, output_path):
    """
    Saves the evaluation report to a text file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\nEvaluation report successfully saved to '{output_path}'")
    except IOError as e:
        print(f"\nError: Could not save report to '{output_path}'. Reason: {e}")


def construct_eval_output_path(original_filepath):
    """
    Constructs the output file path for the evaluation report in the demo directory.
    e.g., 'demo/report_A.txt' -> 'demo/evaluation_report_A.txt'
    """
    base_name = os.path.basename(original_filepath)
    # Remove "synthetic_" prefix if it exists to base the name on the true original
    if base_name.startswith("synthetic_"):
        base_name = base_name[len("synthetic_"):]

    if '.' in base_name:
        file_name, file_ext = base_name.rsplit('.', 1)
        output_filename = f"evaluation_{file_name}.{file_ext}"
    else:
        output_filename = f"evaluation_{base_name}"

    # Assume the output is always saved to the 'demo' directory
    return os.path.join("demo", output_filename)


def main():
    """
    Main function to parse arguments and run the privacy evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the privacy of a synthetic report against an original report."
    )
    parser.add_argument(
        "original_file",
        type=str,
        help="The path to the original report text file.",
    )
    parser.add_argument(
        "synthetic_file",
        type=str,
        help="The path to the synthetic report text file.",
    )
    args = parser.parse_args()

    original_text = read_file_content(args.original_file)
    synthetic_text = read_file_content(args.synthetic_file)

    # --- Generate Report ---
    report_header = [
        "--- Privacy Evaluation Summary ---",
        f"Original file: {args.original_file}",
        f"Synthetic file: {args.synthetic_file}",
        "-" * 34 + "\n"
    ]

    pii_report = analyze_pii_leakage(original_text, synthetic_text)
    text_sim_report = analyze_text_similarity(original_text, synthetic_text)
    lex_sim_report = analyze_lexical_similarity(original_text, synthetic_text)
    sem_sim_report = analyze_semantic_similarity(original_text, synthetic_text)
    stat_report = analyze_statistical_fidelity(original_text, synthetic_text)

    full_report = "\n".join(report_header) + "\n".join([pii_report, text_sim_report, lex_sim_report, sem_sim_report, stat_report])

    # Print the report to the console
    print(full_report)

    # Save the report to a file
    output_path = construct_eval_output_path(args.original_file)
    save_evaluation_report(full_report, output_path)


if __name__ == "__main__":
    main()