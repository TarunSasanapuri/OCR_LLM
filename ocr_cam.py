from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from gtts import gTTS
import tempfile

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini model
model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# Initialize session state variables
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "camera_image" not in st.session_state:
    st.session_state.camera_image = None
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None

# Function to process uploaded images
def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    return None

# Function to extract text using Gemini API
def get_gemini_response(image, prompt):
    response = model.generate_content([image[0], prompt])
    return response.text

def store_text_in_embeddings(texts):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embedding_model)
    vector_store.save_local("faiss_ind")

def translate_text(language, extracted_text):
    prompt = PromptTemplate.from_template(
        "Translate the following text into {language}:\n\n{context}\n\nTranslation:",
        input_variables=["context", "language"]
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": extracted_text, "language": language})
    st.subheader("🌐 Translation:")
    st.write(response)
    audio(response)
    return response

def summary(extracted_text):
    prompt = PromptTemplate.from_template(
        "Provide a concise summary of the following text:\n\n{context}\n\nSummary:",
        input_variables=["context"]
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": extracted_text})
    st.subheader("📝 Summary:")
    st.write(response)
    audio(response)
    return response

def dictionary(word, extracted_text):
    prompt = PromptTemplate.from_template(
        "Extract the meaning of the word '{word}' from the following context:\n\n{context}\n\nMeaning:",
        input_variables=["word", "context"]
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"word": word, "context": extracted_text})
    st.subheader("📖 Dictionary:")
    st.write(response)
    audio(response)
    return response

def audio(text):
    """FIXED: Added missing closing parenthesis"""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            filename = temp_audio.name
            tts.save(filename)
        st.audio(filename, format="audio/mp3")
    except Exception as e:
        st.error(f"Error generating audio: {e}")

# Streamlit App UI
def main():
    st.title("🖊️ MultiLanguage Text Extractor")
    st.header("Extract and Translate Handwritten Text")

    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Upload Image")
        browse = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
        
        # Camera logic
        if st.button("📸 Open
