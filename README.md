# 📰 Advanced AI-Powered Fake News Detector

This repository contains the source code for our **Final Year Project**, an advanced **fake news detection system**.  
The application leverages a **multi-model AI approach** to classify news articles as *Real* or *Fake*, providing a **comparative analysis** of different state-of-the-art techniques.

The system is deployed as an **interactive Streamlit web application** that allows users to test models by pasting text or uploading screenshots of news articles.

---

## ✨ Features

### 🖥️ Interactive Streamlit UI
A clean and user-friendly web interface for real-time predictions.

### 🤖 Multi-Model Analysis
The app doesn’t rely on a single model — it compares results from **three different AI paradigms**:

1. **Classical ML Ensemble** — A collection of five traditional machine learning models.  
2. **Advanced Deep Learning** — A custom-built hybrid neural network with a self-attention mechanism.  
3. **Generative AI (LLM)** — Zero-shot classification using modern Large Language Models (LLMs) that also provide reasoning.

### 🧾 Flexible Input
- **Text Input:** Paste or type news articles directly into the text area.  
- **Image Input:** Upload screenshots of news articles — the app uses **Tesseract OCR** to extract text automatically.

---

## 🖼️ Project Showcase

Here’s a look at the application's main features:

### 🧩 Tab 1: Classical ML Model Comparison
Runs the input through five ML models and displays their predictions and accuracies, providing a clear final verdict.

### 🧠 Tab 2: Advanced Attention Architecture
Shows the prediction from the high-performance deep learning model, its metrics, and a summary of the model’s architecture.

### 🗣️ Tab 3: Generative AI (LLM) Inference
Compares verdicts from **Llama**, **OpenAI**, and **Gemini**, each with a human-readable rationale for its decision.

### 🖋️ Image-to-Text (OCR) Functionality
Extracts text from uploaded images using Tesseract OCR and feeds it into the models for classification.
---

## 🤖 Models & Architecture

This project combines **three complementary approaches** for fake news detection, accessible via the three tabs in the Streamlit app.

---

### **1️⃣ Classical Machine Learning Models**

#### 🧹 Preprocessing
- Text cleaning and stemming using `PorterStemmer`
- Vectorization using `TfidfVectorizer`

#### 🧠 Models Trained
- Decision Tree Classifier  
- Gradient Boosting Classifier  
- Linear SVC  
- Logistic Regression  
- Random Forest Classifier  

#### 📈 Performance
Each model’s accuracy is displayed in the app (e.g., Decision Tree ≈ 99.6%, Gradient Boost ≈ 99.5%).

> Training details available in **`multimodel_training.ipynb`**.

---

### **2️⃣ Advanced Attention Architecture**

A custom **deep learning model** designed to capture both **n-gram patterns** and **contextual relationships** in text.

#### ⚙️ Preprocessing
- Cleaned text tokenized via Keras `Tokenizer`
- Padding applied (MAX_LEN = 60)

#### 🧩 Hybrid Model Structure
- **Embedding Layer:** Converts tokens into 128-dimension vectors  
- **Branch 1 (N-gram):** Conv1D + GlobalMaxPooling1D  
- **Branch 2 (Sequential Context):** BiLSTM + AttentionLayer  
- **Concatenation → Dense Layers → Output**

#### 📊 Performance
| Metric | Score |
|:-------|:------:|
| Accuracy | 0.9996 |
| Precision | 0.9994 |
| Recall | 0.9997 |
| F1-Score | 0.9995 |
| ROC AUC | 1.0000 |

> Training details available in **`attention_framework.ipynb`**.

---

### **3️⃣ Generative AI (LLM) Inference**

Explores **Large Language Models** as zero-shot fact-checkers.  
Each LLM provides a verdict *and* an explanation.

#### 🧩 Prompting Strategy
Models are instructed to act as expert fact-checkers, evaluating:
- Tone and sensationalism  
- Logical consistency  
- Factual alignment  

#### 🧠 Models Used
- **Llama 3** (via Groq)  
- **OpenAI OSS Model** (via Groq)  
- **Gemini 2.5 Flash** (via Google GenAI)

> Logic handled in **`llm_inference.py`** — includes structured output parsing.

---

## 🚀 Getting Started

### 🧩 1. Prerequisites
- Python 3.9+  
- Git  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract): Required for OCR functionality

#### 📘 Tesseract Installation
Make sure `tesseract` is installed and available in your system PATH.

---

### ⚙️ 2. Clone the Repository
```bash
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
```

### 📦 3. Install Dependencies
Use a virtual environment for best practice.
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader stopwords
```

### 💻 5. Run the Application
```bash
streamlit run app.py
```

### 🧭 Usage
- Enter a news article or upload an image.
- Click 🔎 Predict.
- View results across all three model tabs.
