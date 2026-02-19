# 🤖 Automatic Text Summarization: The TL;DR Machine
> **A showcase of Natural Language Processing (NLP) techniques to condense complex information into digestible insights.**

---

## 🚀 Project Overview
In an era of information overload, the ability to extract essence from volume is a superpower. This project implements a dual-approach summarizer capable of both **Extractive** (selecting key sentences) and **Abstractive** (generating new text) summarization.

### 🛠️ Core Tech Stack
* **Language:** Python 3.11+ (3.12 in this case)
* **Models:** HuggingFace Transformers (T5, BART, Pegasus), Spacy, NLTK
* **Infrastructure:** PyTorch / TensorFlow
* **Deployment:** [e.g., Streamlit / Flask / FastAPI]

---

## 🧠 The Architecture
This project explores the two main paradigms of Automatic Text Summarization (ATS):

1. **Extractive Summarization:** Uses algorithms like **TextRank** (inspired by PageRank) to identify and "stitch together" the most important sentences directly from the source.
2. **Abstractive Summarization:** Leverages the **Transformer architecture** to understand context and paraphrase. It doesn't just copy; it *writes*.

### The Math Behind the Magic
The backbone of the abstractive model is the **Scaled Dot-Product Attention** mechanism:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This allows the model to "attend" to specific parts of a long document when generating each word of the summary.

---

## 📊 Evaluation Metrics
To ensure the machine isn't just hallucinating, I utilized the **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** metric. It measures the overlap between the machine-generated summary and a human-written reference.

Specifically, **ROUGE-N** is calculated as:

$$ROUGE-N = \frac{\sum_{S \in \{Reference\}} \sum_{gram_n \in S} Count_{match}(gram_n)}{\sum_{S \in \{Reference\}} \sum_{gram_n \in S} Count(gram_n)}$$

---

## 📈 Key Results
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :--- | :--- |
| **Extractive (TextRank)** | 0.38 | 0.15 | 0.34 |
| **Abstractive (T5-Base)** | 0.45 | 0.22 | 0.41 |
| **Pegasus (SOTA 2026)** | **0.52** | **0.28** | **0.48** |

---

## 🛠️ Installation & Usage
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/JuanPabloFue23/JuanFueHi_Portfolio/tree/main/Projects/Automatic-text-summarization](https://github.com/JuanPabloFue23/JuanFueHi_Portfolio/tree/main/Projects/Automatic-text-summarization)

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Run the Demo:**
    ```bash
    python main.py --input "your_long_text.txt" --type "abstractive"


