# ----------------------
# Imports 
# -----------------------

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle, os
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
from load_utils import *
import pandas as pd
import pytesseract
from keras.models import load_model
from llm_inference import *

# -----------------------
# Streamlit UI Setup
# -----------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #F9FAFB;
        }
        .title {
            text-align: center;
            color: #2B547E;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 1rem;
            color: #5A5A5A;
        }
        .stTabs [data-baseweb="tab-list"] {
            justify-content: space-evenly;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem;
            font-weight: 600;
        }
        .result-box {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/7.0.1/css/all.min.css" integrity="sha512-2SwdPD6INVrV/lHTZbO2nodKhrnDdJK9/kg2XD1r9uGqPo1cUbujc+IYdlYdEErWNu69gVcYgdxlmVmzTWnetw==" crossorigin="anonymous" referrerpolicy="no-referrer" />
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📰 Advanced Fake vs Real News Classifier</div>', unsafe_allow_html=True)
st.markdown('''<div class="subtitle">Artificial Intelligence powered by  
            <i class="fa-solid fa-square-binary"></i>
            <i class="fa-solid fa-hexagon-nodes-bolt"></i>
            </div>''', unsafe_allow_html=True)

user_input = st.text_area("Enter News Text / Image URL..", placeholder="Type or paste a news chunk over here...", height=120)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("### 📚 About the Model")
    st.markdown("""

    **Model Architecture Highlights:**
    - Embedding Layer (128 dimensions)
    - Conv1D Layer for local feature extraction
    - Bidirectional LSTM for sequential learning
    - Custom Attention Layer for context focus
    - Dense layers for classification
    - Trained on cleaned, labeled news dataset
    """)

with col2:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], )
    
    predict_btn = st.button("🔎 Predict", use_container_width=True)

with col3:
    st.markdown("### ⚙️ Key Features")
    st.markdown("""
    - 🧹 Text preprocessing and cleaning  
    - 🧠 Tokenization and padding  
    - 💡 Attention-enhanced deep neural network  
    - 🔁 Saved model and tokenizer for reproducibility  
    - 💻 Streamlit UI for real-time predictions
    """)


# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Trivial ML Model's", "📊 Advanced Attention Architecture", "📊Generative AI Predictions."])
# -----------------------
# Tab 1: News Prediction using trivial machine learning models.
# -----------------------
with tab1:
    
    # There can be 3 conditions
    # 1. Button is pressed with a news chunk in it.
    # 2. Button is pressed with a news image attached to it.
    # 3. Button is pressed but no image is attached or no text is written.

    if predict_btn and user_input.strip() != "":

        # Loading model's and vectorizer.
        model_dt, model_gb, model_svc, model_lr, model_rf = load_pretrained_models()
        text_vectorizer = load_pretrained_text_vectorizer()

        # Text Preprocessing.
        processed_text = pre_process_text(user_input)
        print(processed_text)
        processed_text = text_vectorizer.transform([processed_text])
        
        # Cattering the model names and their accuracy.
        model_info = {"Model Name": [], "Model Accuracy": [], "Model Prediction":[]}

        with open('models/model_stats.txt', 'r') as file:
            s = file.readlines()

            for i in s:
                e = i.split()
                model_accuracy = round(float(e[-1]) * 100, 3)
                model_name = ' '.join(e[:-1])

                model_info['Model Name'].append(model_name)
                model_info['Model Accuracy'].append(model_accuracy)
                
        model_info['Model Prediction'].append(model_dt.predict(processed_text)[0])
        model_info['Model Prediction'].append(model_gb.predict(processed_text)[0])
        model_info['Model Prediction'].append(model_svc.predict(processed_text)[0])
        model_info['Model Prediction'].append(model_lr.predict(processed_text)[0])
        model_info['Model Prediction'].append(model_rf.predict(processed_text)[0])

        df = pd.DataFrame(model_info)
        df['Model Prediction'] = df['Model Prediction'].apply(label_conv)
        
        # Model's Final Prediction..
        st.subheader("🔍 Model Predictions:")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Decision Tree", df.iloc[0, 2])
        col2.metric("Gradient Boost", df.iloc[1, 2])
        col3.metric("Linear SVC", df.iloc[2, 2])
        col4.metric("Logistic Regression", df.iloc[3, 2])
        col5.metric("Random Forest", df.iloc[4, 2])

        # Model Accuracy Score.
        st.subheader("📈 Model Accuracy Scores:")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Decision Tree", df.iloc[0, 1])
        col2.metric("Gradient Boost", df.iloc[1, 1])
        col3.metric("Linear SVC", df.iloc[2, 1])
        col4.metric("Logistic Regression", df.iloc[3, 1])
        col5.metric("Random Forest", df.iloc[4, 1])

        # Final Prediction Label.
        results = df['Model Prediction'].tolist()
        final_output = max(set(results), key=results.count)
        st.success(f"✅ Final Verdict: **{final_output.upper()}**")
        
    elif predict_btn and (uploaded_file is not None):

        pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"

        # Load image from a URL.
        image = Image.open(uploaded_file)
        col21, col22 = st.columns([1, 2])

        extracted_text = pytesseract.image_to_string(image)

        with col21:
            # Display the image
            st.write("")
            st.image(image, caption="Uploaded Image", width=300)
        
        with col22:
            st.title("♘ Extracted Text from the Image..")

            if (extracted_text != ""):
                st.write(extracted_text)
            
            else:
                st.write("The Image has no text attribute to be displayed.")
    
        if (extracted_text != ""):

            # Loading model's and vectorizer.
            model_dt, model_gb, model_svc, model_lr, model_rf = load_pretrained_models()
            text_vectorizer = load_pretrained_text_vectorizer()

            # Text Preprocessing.
            processed_text = pre_process_text(extracted_text)
            print(processed_text)
            processed_text = text_vectorizer.transform([processed_text])
            
            # Cattering the model names and their accuracy.
            model_info = {"Model Name": [], "Model Accuracy": [], "Model Prediction":[]}

            with open('models/model_stats.txt', 'r') as file:
                s = file.readlines()

                for i in s:
                    e = i.split()
                    model_accuracy = round(float(e[-1]) * 100, 3)
                    model_name = ' '.join(e[:-1])

                    model_info['Model Name'].append(model_name)
                    model_info['Model Accuracy'].append(model_accuracy)
                    
            model_info['Model Prediction'].append(model_dt.predict(processed_text)[0])
            model_info['Model Prediction'].append(model_gb.predict(processed_text)[0])
            model_info['Model Prediction'].append(model_svc.predict(processed_text)[0])
            model_info['Model Prediction'].append(model_lr.predict(processed_text)[0])
            model_info['Model Prediction'].append(model_rf.predict(processed_text)[0])

            df = pd.DataFrame(model_info)
            df['Model Prediction'] = df['Model Prediction'].apply(label_conv)
            
            # Model's Final Prediction..
            st.subheader("🔍 Model Predictions:")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Decision Tree", df.iloc[0, 2])
            col2.metric("Gradient Boost", df.iloc[1, 2])
            col3.metric("Linear SVC", df.iloc[2, 2])
            col4.metric("Logistic Regression", df.iloc[3, 2])
            col5.metric("Random Forest", df.iloc[4, 2])

            # Model Accuracy Score.
            st.subheader("📈 Model Accuracy Scores:")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Decision Tree", df.iloc[0, 1])
            col2.metric("Gradient Boost", df.iloc[1, 1])
            col3.metric("Linear SVC", df.iloc[2, 1])
            col4.metric("Logistic Regression", df.iloc[3, 1])
            col5.metric("Random Forest", df.iloc[4, 1])

            # Final Prediction Label.
            results = df['Model Prediction'].tolist()
            final_output = max(set(results), key=results.count)
            st.success(f"✅ Final Verdict: **{final_output.upper()}**")
        
        else:
            st.warning("Please enter a image with proper text, in order to start with the news prediction.")

    elif predict_btn:
        st.warning("Please enter a proper chunk of news or upload an image before prediction.")

# -----------------------
# Tab 2: News Prediction using Advanced Neural Network Architecture.
# -----------------------
with tab2:
    
    if (predict_btn) and (user_input.strip() != ""):
        model = load_model(os.path.join("saved_model", "best_model.keras"), custom_objects={"AttentionLayer": AttentionLayer})

        with open('saved_model/tokenizer.pkl', 'rb') as file:
            tk = pickle.load(file)

        preds, probs = predict_texts([user_input.strip()], model, tk)
        st.success(f"✅ Final Verdict: **{preds.upper()}**")

        st.markdown("### 📊 Model Performance.")
        print_model_info()

        st.title("🧩 Model & Parameter Summary Table")
        pretty_model_summary(model)


    elif (predict_btn) and (uploaded_file is not None):
        model = load_model(os.path.join("saved_model", "best_model.keras"), custom_objects={"AttentionLayer": AttentionLayer})

        with open('saved_model/tokenizer.pkl', 'rb') as file:
            tk = pickle.load(file)

        # Extract information/text from the Image.
        pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"

        # Load image from a URL.
        image = Image.open(uploaded_file)
    
        col11, col12 = st.columns([1, 2])
        extracted_text = pytesseract.image_to_string(image)

        with col11:
            # Display the image
            st.write("")
            st.image(image, caption="Uploaded Image", width=300)
        
        with col12:

            st.title("♘ Extracted Text from the Image..")

            if (extracted_text != ""):
                st.write(extracted_text)
            
            else:
                st.write("The Image has no text attribute to be displayed.")

        if (extracted_text != ""):
            preds, probs = predict_texts([extracted_text], model, tk)
            st.success(f"✅ Final Verdict: **{preds.upper()}**")

            st.markdown("### 📊 Model Performance.")
            print_model_info()

            st.title("🧩 Model & Parameter Summary Table")
            pretty_model_summary(model)

        else:
            st.warning("Please enter a image with proper text, in order to start with the news prediction.")


    elif predict_btn:
        st.warning("Please enter a proper chunk of news or upload an image before prediction.")


# -----------------------
# Tab 3: News Prediction using Different Top-Tier LLM Models.
# -----------------------
with tab3:
    
    if (predict_btn) and (user_input.strip() != ""):
        
        user_input = user_input.strip()
        col31, col32, col33 = st.columns(3)

        with col31:
            try:

                st.markdown('''<p style="font-size:30px;">Llama's Inference.</p>''', unsafe_allow_html=True)
                llama_result = dict(classify_news_using_llm(llama_model, user_input))
                st.success(f"✅ Final Verdict: **{llama_result['news_type'].upper()}**")

                st.title("Reasoning Behind Llama's Decision..")
                st.write(llama_result['reason'])

            except Exception as e:
                st.warning(f"Problem During Execution : {e}")

        with col32:
            try:

                st.markdown('''<p style="font-size:30px;">OpenAI's Inference.</p>''', unsafe_allow_html=True)
                openai_result = dict(classify_news_using_llm(openai_model, user_input))
                st.success(f"✅ Final Verdict: **{openai_result['news_type'].upper()}**")

                st.title("Reasoning Behind OpenAI's Decision..")
                st.write(openai_result['reason'])

            except Exception as e:
                st.warning(f"Problem During Execution : {e}")

        with col33:
            try:

                st.markdown('''<p style="font-size:30px;">Gemini's Inference.</p>''', unsafe_allow_html=True)
                gemini_result = dict(classify_news_using_llm(gemini_model, user_input))
                st.success(f"✅ Final Verdict: **{gemini_result['news_type'].upper()}**")

                st.title("Reasoning Behind Gemini's Decision..")
                st.write(gemini_result['reason'])

            except Exception as e:
                st.warning(f"Problem During Execution : {e}")


    elif (predict_btn) and (uploaded_file is not None):
        
        # Extract information/text from the Image.
        pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"

        # Load image from a URL.
        image = Image.open(uploaded_file)
    
        col11, col12 = st.columns([1, 2])
        extracted_text = pytesseract.image_to_string(image)

        with col11:
            # Display the image
            st.write("")
            st.image(image, caption="Uploaded Image", width=300)
        
        with col12:

            st.title("♘ Extracted Text from the Image..")

            if (extracted_text != ""):
                st.write(extracted_text)
            
            else:
                st.write("The Image has no text attribute to be displayed.")

        col31, col32, col33 = st.columns(3)

        if (extracted_text != ""):
            with col31:
                try:

                    st.markdown('''<p style="font-size:30px;">Llama's Inference.</p>''', unsafe_allow_html=True)
                    llama_result = dict(classify_news_using_llm(llama_model, user_input))
                    st.success(f"✅ Final Verdict: **{llama_result['news_type'].upper()}**")

                    st.title("Reasoning Behind Llama's Decision..")
                    st.write(llama_result['reason'])

                except Exception as e:
                    st.warning(f"Problem During Execution : {e}")

            with col32:
                try:

                    st.markdown('''<p style="font-size:30px;">OpenAI's Inference.</p>''', unsafe_allow_html=True)
                    openai_result = dict(classify_news_using_llm(openai_model, user_input))
                    st.success(f"✅ Final Verdict: **{openai_result['news_type'].upper()}**")

                    st.title("Reasoning Behind OpenAI's Decision..")
                    st.write(openai_result['reason'])

                except Exception as e:
                    st.warning(f"Problem During Execution : {e}")

            with col33:
                try:

                    st.markdown('''<p style="font-size:30px;">Gemini's Inference.</p>''', unsafe_allow_html=True)
                    gemini_result = dict(classify_news_using_llm(gemini_model, user_input))
                    st.success(f"✅ Final Verdict: **{gemini_result['news_type'].upper()}**")

                    st.title("Reasoning Behind Gemini's Decision..")
                    st.write(gemini_result['reason'])

                except Exception as e:
                    st.warning(f"Problem During Execution : {e}")
        
        else:
            st.warning("Please enter a image with proper text, in order to start with the news prediction.")


    elif (predict_btn):
        st.warning("Please enter a proper chunk of news or upload an image before prediction.")

    
