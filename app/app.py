"""
Arabic NLP System - Streamlit Application

This application combines:
- Arabic question answering using a fine-tuned Qwen model
- Arabic text classification using a fine-tuned AraBERT model
- Arabic-to-English translation through the OpenRouter API
"""

import streamlit as st
import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from openai import OpenAI


# Load trained classification model and label encoder
classification_model = AutoModelForSequenceClassification.from_pretrained("best_classification_bert/best_classification_bert")
classification_tokenizer = AutoTokenizer.from_pretrained("best_classification_bert/best_classification_bert")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

classification_model = classification_model.float()
classification_model.eval()


# Load trained question-answering model
qa_tokenizer = AutoTokenizer.from_pretrained("best_qa_qwen_merged/best_qa_qwen_merged")
qa_model = AutoModelForCausalLM.from_pretrained("best_qa_qwen_merged/best_qa_qwen_merged")

qa_model = qa_model.float()
qa_model.eval()


# Configure OpenRouter client for Arabic-to-English translation
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Translate Arabic text into English
def translate_to_english(text):

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional translator. Translate Arabic text into English only."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        max_tokens=1000,
        temperature=0.3
    )

    return response.choices[0].message.content

# Predict the category of the Arabic question-answer pair
def classify_question(question, answer):

    text = str(question) + " " + str(answer)

    inputs = classification_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = classification_model(**inputs)
    predicted_id = outputs.logits.argmax(dim=1).item()
    category = label_encoder.inverse_transform([predicted_id])[0]
    return category


# Generate an Arabic answer using the fine-tuned Qwen model
def generate_answer(question):

    prompt = f"Answer the Arabic question:\nQuestion: {question}\nAnswer:"

    inputs = qa_tokenizer(prompt, return_tensors="pt", truncation=True,  max_length=256)

    with torch.no_grad():
        outputs = qa_model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=qa_tokenizer.eos_token_id)

    result = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = result.split("Answer:")[-1].strip()

    return answer



# Streamlit user interface
st.set_page_config(page_title="Arabic NLP System")

st.title("Welcome to Arabic NLP System")
st.write("This system classifies Arabic questions, generates answers, and translates outputs to English.")

user_question = st.text_input("Enter your Arabic question:")

if st.button("Generate Results"):

    if user_question.strip() == "":
        st.warning("Please enter a question.")

    else:
        with st.spinner("Generating answer..."):
            generated_answer = generate_answer(user_question)

        with st.spinner("Classifying question..."):
            predicted_category = classify_question(user_question, generated_answer)

        with st.spinner("Translating outputs..."):

            translated_question = translate_to_english(user_question)
            translated_answer = translate_to_english(generated_answer)
            translated_category = translate_to_english(predicted_category)

        st.subheader("Original Question")
        st.write(user_question)

        st.subheader("Question Translation")
        st.write(translated_question)

        st.divider()

        st.subheader("Predicted Category:")
        st.write(predicted_category)

        st.subheader("Category Translation:")
        st.write(translated_category)

        st.divider()

        st.subheader("Generated Answer:")
        st.write(generated_answer)

        st.subheader("Answer Translation:")
        st.write(translated_answer)


        

