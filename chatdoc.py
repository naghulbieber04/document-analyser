import streamlit as st
import PyPDF2
import nltk
import spacy
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

# Load question answering pipeline
qa_pipeline = pipeline("question-answering")

# Extract text from PDF or TXT file
def extract_text(file):
    text = ""
    if file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    elif file.name.endswith('.txt'):
        text = file.read().decode('utf-8')
    return text

# Analyze the document
# Removed cache_data to avoid NLTK resource caching issues
def analyze_document(text):
    nltk.download('punkt')  # Ensure proper punkt model
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    mood = "Positive 😊" if sentiment > 0 else ("Negative 😞" if sentiment < 0 else "Neutral 😐")

    freq_dist = nltk.FreqDist(filtered_words)
    entities = [(ent.text, ent.label_) for ent in nlp_spacy(text).ents]

    return {
        "sentences": sentences,
        "word_count": len(filtered_words),
        "sentence_count": len(sentences),
        "top_words": freq_dist.most_common(10),
        "sentiment": sentiment,
        "mood": mood,
        "entities": entities
    }

# Main Streamlit app
st.set_page_config(page_title="Document QA Chatbot", page_icon="📄")
st.title("📄 Document QA Chatbot")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    doc_text = extract_text(uploaded_file)
    if not doc_text.strip():
        st.warning("The document appears to be empty or unreadable.")
    else:
        analysis = analyze_document(doc_text)

        st.subheader("📊 Document Summary")
        st.write(f"**Total Words:** {analysis['word_count']}")
        st.write(f"**Total Sentences:** {analysis['sentence_count']}")
        st.write(f"**Sentiment:** {analysis['mood']} (Score: {analysis['sentiment']:.2f})")

        st.write("\n**Top 10 Frequent Words:**")
        for word, freq in analysis['top_words']:
            st.write(f"{word}: {freq} times")

        st.write("\n**Named Entities:**")
        if analysis['entities']:
            for text, label in analysis['entities']:
                st.write(f"{text} → {label}")
        else:
            st.write("No named entities found.")

        st.subheader("💬 Ask Questions About the Document")
        user_question = st.text_input("Type your question:")

        if user_question:
            try:
                result = qa_pipeline(question=user_question, context=doc_text)
                st.success(f"Answer: {result['answer']}")
            except Exception as e:
                st.error(f"❌ Error answering the question: {e}")
