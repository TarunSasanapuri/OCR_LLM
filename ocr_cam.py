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
import os

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini model
model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")  # Updated model name
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
if "faiss_ready" not in st.session_state:
    st.session_state.faiss_ready = False

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
    st.subheader("Translation:")
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
    st.subheader("Summary:")
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
    st.subheader("Dictionary:")
    st.write(response)
    audio(response)
    return response

def audio(text):
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
        browse = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"], key="file_uploader")
        
        # Camera logic
        if st.button("📸 Open Camera", key="camera_btn"):
            st.session_state.camera_mode = True
        
        if "camera_mode" not in st.session_state:
            st.session_state.camera_mode = False
            
        if st.session_state.camera_mode:
            capture = st.camera_input("Capture an image", key="camera_input")
            if capture is not None:
                st.session_state.camera_image = capture
                st.session_state.uploaded_file = None
                st.session_state.camera_mode = False
                st.rerun()

        # Store uploaded file
        if browse is not None:
            st.session_state.uploaded_file = browse
            st.session_state.camera_image = None
            st.rerun()

        st.divider()
        
        # Action buttons - only show if text extracted
        if st.session_state.extracted_text:
            with st.expander("📝 SUMMARY"):
                if st.button("Get Summary", key="summary_btn"):
                    summary(st.session_state.extracted_text)
            
            with st.expander("🌐 TRANSLATE"):
                language = st.text_input("Enter target language (e.g., Spanish, Hindi):", key="lang_input")
                if st.button("Translate", key="translate_btn") and language:
                    translate_text(language, st.session_state.extracted_text)
                elif st.button("Translate", key="translate_btn"):
                    st.error("Please enter a language!")
            
            with st.expander("📖 DICTIONARY"):
                word = st.text_input("Enter word to lookup:", key="word_input")
                if st.button("Get Meaning", key="dict_btn") and word:
                    dictionary(word, st.session_state.extracted_text)
                elif st.button("Get Meaning", key="dict_btn"):
                    st.error("Please enter a word!")

    # Main content
    image = None
    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    elif st.session_state.camera_image is not None:
        image = Image.open(st.session_state.camera_image)
        st.image(image, caption="Captured Image", use_container_width=True)
    else:
        st.warning("👆 Please upload or capture an image first!")

    # Extract button
    if st.button("🔍 Extract Text", type="primary"):
        if image is None:
            st.error("❌ Please upload or capture an image first!")
        else:
            with st.spinner("Extracting text..."):
                try:
                    image_data = input_image_details(
                        st.session_state.uploaded_file or st.session_state.camera_image
                    )
                    if image_
                        input_prompt = """You are an expert in understanding handwritten text. 
                        Extract the complete text from the uploaded image accurately."""
                        
                        response = get_gemini_response(image_data, input_prompt)
                        store_text_in_embeddings([response])
                        st.session_state.extracted_text = response
                        st.session_state.faiss_ready = True
                        
                        st.success("✅ Text extracted successfully!")
                        st.subheader("📄 Extracted Text:")
                        st.write(response)
                        audio(response)
                    else:
                        st.error("❌ Could not process image data.")
                except Exception as e:
                    st.error(f"❌ Error extracting text: {str(e)}")

    # Show extracted text status
    if st.session_state.extracted_text:
        st.success(f"✅ Ready for processing! Text length: {len(st.session_state.extracted_text)} chars")
    else:
        st.info("👆 Extract text first to enable summary, translation, and dictionary features.")

if __name__ == "__main__":
    main()
