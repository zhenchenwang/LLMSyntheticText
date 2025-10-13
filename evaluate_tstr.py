"""
This script performs a 'Train on Synthetic, Test on Real' (TSTR) evaluation
to measure the utility of the synthetic data.

Methodology:
1.  Load the 'real' and 'synthetic' training datasets from the 'tstr_data' directory.
2.  Load the 'real' test dataset.
3.  Vectorize the text data using TF-IDF.
4.  Train a Logistic Regression classifier on the 'real' training data.
5.  Train an identical Logistic Regression classifier on the 'synthetic' training data.
6.  Evaluate both models on the 'real' test set.
7.  Compare their F1-scores to determine the performance gap and, thus, the utility
    of the synthetic data.
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

"""
This script performs a 'Train on Synthetic, Test on Real' (TSTR) evaluation
to measure the utility of the synthetic data.

Methodology:
1.  Load the 'real' and 'synthetic' training datasets from the 'tstr_data' directory.
2.  Load the 'real' test dataset.
3.  Vectorize the text data using TF-IDF.
4.  Train a Logistic Regression classifier on the 'real' training data.
5.  Train an identical Logistic Regression classifier on the 'synthetic' training data.
6.  Evaluate both models on the 'real' test set.
7.  Compare their F1-scores to determine the performance gap and, thus, the utility
    of the synthetic data.
"""

def load_data_from_directory(directory):
    """Loads text files and their labels from subdirectories."""
    texts = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for fname in os.listdir(label_path):
                with open(os.path.join(label_path, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
    return shuffle(texts, labels, random_state=42)

def run_tstr_evaluation():
    """Main function to run the TSTR evaluation."""

    # 1. Load Data
    print("Loading datasets...")
    real_train_texts, real_train_labels = load_data_from_directory("tstr_data/real")
    synth_train_texts, synth_train_labels = load_data_from_directory("tstr_data/synthetic")
    real_test_texts, real_test_labels = load_data_from_directory("tstr_data/test")

    print(f"Real training samples: {len(real_train_texts)}")
    print(f"Synthetic training samples: {len(synth_train_texts)}")
    print(f"Real test samples: {len(real_test_texts)}\n")

    # 2. Vectorize Data
    vectorizer = TfidfVectorizer(max_features=1000)
    X_real_train = vectorizer.fit_transform(real_train_texts)
    X_synth_train = vectorizer.transform(synth_train_texts)
    X_real_test = vectorizer.transform(real_test_texts)

    # 3. Train Models
    print("Training models...")
    model_real = LogisticRegression(random_state=42)
    model_real.fit(X_real_train, real_train_labels)

    model_synth = LogisticRegression(random_state=42)
    model_synth.fit(X_synth_train, synth_train_labels)
    print("Training complete.\n")

    # 4. Evaluate Models
    print("Evaluating models on the real test set...")
    preds_real = model_real.predict(X_real_test)
    preds_synth = model_synth.predict(X_real_test)

    f1_real = f1_score(real_test_labels, preds_real, pos_label='device_issue')
    f1_synth = f1_score(real_test_labels, preds_synth, pos_label='device_issue')

    # 5. Report Results
    print("\n--- TSTR Evaluation Results ---")
    print(f"F1 Score (Real Data Model):   {f1_real:.4f}")
    print(f"F1 Score (Synthetic Data Model): {f1_synth:.4f}")
    print("---------------------------------")

    performance_gap = abs(f1_real - f1_synth)
    print(f"\nPerformance Gap (Difference in F1 scores): {performance_gap:.4f}")

    if performance_gap < 0.1:
        print("Conclusion: Excellent utility. The synthetic data is a strong substitute for real data.")
    elif performance_gap < 0.25:
        print("Conclusion: Good utility. The synthetic data retains significant value.")
    else:
        print("Conclusion: Limited utility. The synthetic data does not capture the real data's properties well.")


if __name__ == "__main__":
    run_tstr_evaluation()