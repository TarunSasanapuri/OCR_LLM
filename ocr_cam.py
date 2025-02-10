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

# Function to create FAISS vector database
def store_text_in_embeddings(texts):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embedding_model)
    vector_store.save_local("faiss_file") 

def extract_text(input):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
# -> Load FAISS vector store
    vector_store = FAISS.load_local("faiss_file", embedding_model, allow_dangerous_deserialization=True)
# -> Perform similarity search
    retrieved_texts = vector_store.similarity_search(input)  
    if not retrieved_texts:
        st.write("No text found for translation.")
        return
    extracted_text = retrieved_texts[0].page_content
    return extracted_text


# Function to translate extracted text

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
        uploaded_file = st.camera_input("Capture")
        summary_btn=st.button("Get Summary")
        language = st.text_input("Enter the language to translate the extracted text into:")
        translate_button = st.button("Translate Text")
        word=st.text_input("Enter the word to know its meaning ")
        dict_btn=st.button("Dictionary")
    
    image = ""

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    extract_button = st.button("Extract Text")
    input_prompt = """You are an expert in understanding handwritten text. Extract the complete text from the uploaded image."""

    if extract_button:
        if uploaded_file is None:
            st.error("Please upload an image first!")
        else:
            image_data = input_image_details(uploaded_file)
            response = get_gemini_response(image_data, input_prompt)
            store_text_in_embeddings([response])
            st.subheader("Extracted Text:")
            st.write(response)


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
if __name__=="__main__":
    main()