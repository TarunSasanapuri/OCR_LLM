from dotenv import load_dotenv
load_dotenv()  ## Load all the environment variables from .env

import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini model
model = genai.GenerativeModel(model_name="gemini-2.0-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize session state variables if not already defined
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
        bytes_data = uploaded_file.getvalue()  # Read file bytes
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to extract text using Gemini API
def get_gemini_response(image, prompt):
    response = model.generate_content((image[0], prompt))
    return response.text

def store_text_in_embeddings(texts):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embedding_model)
    vector_store.save_local("faiss_ind") 

def extract_text(input):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
# -> Load FAISS vector store
    vector_store = FAISS.load_local("faiss_ind", embedding_model, allow_dangerous_deserialization=True)
# -> Perform similarity search
    retrieved_texts = vector_store.similarity_search(input)  
    if not retrieved_texts:
        st.write("No text found for translation.")
        return
    extracted_text = retrieved_texts[0].page_content
    return extracted_text

def translate_text(input,extracted_text):
    prompt_template = PromptTemplate(
        template="Translate the following text into {language}:\n\n{context}\n\nTranslation:",
        input_variables=["context", "language"]
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(context=extracted_text, language=input)
    st.subheader("Translation:")
    st.write(response)

# Function to make context into summary
def summary(extracted_text):
    prompt_template = PromptTemplate(
        template="get summary for the {context}\n\nSummary:",
        input_variables=["context"]
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(context=extracted_text)
    st.subheader("Summary:")
    st.write(response)

# Function for Dictionary
def dictionary(word,extracted_text):
    prompt_template=PromptTemplate(
        template="Get the meaning of the {word} from the {context}\n\nMeaning:",
        input_variables=["word","context"]
    )
    chain=LLMChain(llm=llm,prompt=prompt_template)
    response=chain.run(context=extracted_text,word=word)
    st.subheader("Dictionary")
    st.write(response)

# Streamlit App UI
def main():
    st.title("MultiLanguage Text Extractor")
    st.header("Extract and Translate Handwritten Text")

    with st.sidebar:
        browse = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])
        capture = st.camera_input("Capture an image")

        # Store image when uploaded or captured
        if browse is not None:
            st.session_state.uploaded_file = browse
            st.session_state.camera_image = None  # Reset camera input

        if capture is not None:
            st.session_state.camera_image = capture
            st.session_state.uploaded_file = None  # Reset uploaded file

        summary_btn = st.button("Get Summary")
        language = st.text_input("Enter the language to translate the extracted text into:")
        translate_button = st.button("Translate Text")
        word = st.text_input("Enter the word to know its meaning")
        dict_btn = st.button("Dictionary")

    image = None  # Initialize image variable

    # Check if an image is stored in session state and display it
    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    elif st.session_state.camera_image is not None:
        image = Image.open(st.session_state.camera_image)
        st.image(image, caption="Captured Image", use_container_width=True)

    else:
        st.warning("No image uploaded or captured!")

    extract_button = st.button("Extract Text")
    input_prompt = """You are an expert in understanding handwritten text. Extract the complete text from the uploaded image."""

    if extract_button:
        if image is None:
            st.error("Please upload or capture an image first!")
        else:
            try:
                # Get image details
                image_data = input_image_details(
                    st.session_state.uploaded_file or st.session_state.camera_image
                )
                st.session_state.image_data = image_data

                # Extract text
                response = get_gemini_response(image_data, input_prompt)
                store_text_in_embeddings([response])
                st.session_state.extracted_text = response

                st.subheader("Extracted Text:")
                st.write(response)

            except Exception as e:
                st.error(f"Error extracting text: {e}")
    if translate_button:
        if not language:
            st.error("Please enter a language for translation!")
        else:
            Extract_text=extract_text(language)
            translate_text(language,Extract_text)


    if summary_btn:
        try:
            Extract_text=extract_text('summary')
            summary(Extract_text)
        except Exception as e:
            st.write("Get the context first")


    if dict_btn and word:
        if not word:
            st.error("Please enter a word to know it's meaning!")
        else:
            Extract_text=extract_text(word)
            dictionary(word,Extract_text)
if __name__ == "__main__":
    main()
