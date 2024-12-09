import json
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import HumanMessage, AIMessage
import os
import fitz  # PyMuPDF

# Load the API key from secrets.toml
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets["LLAMA_API_KEY"]

st.set_page_config(page_title="Document Genie", layout="wide")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []  
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def load_customer_data():
    with open("data\\customer_data.json", "r") as f:
        return json.load(f)

def get_openai_model(temperature=0.3):
    return ChatOpenAI(model_name="gpt-4", temperature=temperature, openai_api_key=OPENAI_API_KEY)

def extract_text_from_pdf(pdf_file):
    # Convert the uploaded PDF file to a file-like object
    pdf_bytes = pdf_file.read()
    
    # Open the file-like object using PyMuPDF (fitz)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Extract text from all pages
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load each page
        text += page.get_text()  # Extract text from each page
    
    return text

def load_and_process_pdf(pdf_file):
    # Extract text using PyMuPDF
    text = extract_text_from_pdf(pdf_file)
    
    if not text:
        st.error("No text extracted from the PDF. Please check the document.")
        return None
    
    # Print first 500 characters of the extracted text (for debugging)
    print(f"Extracted Text: {text[:500]}")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Debugging: Check the number of chunks
    print(f"Number of chunks created: {len(chunks)}")
    
    if len(chunks) == 0:
        st.error("No chunks created from the document. Please check the input PDF.")
        return None
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    try:
        embedded_texts = embeddings.embed_documents(chunks)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None
    
    # Debugging: Check the embeddings
    if not embedded_texts:
        st.error("Embeddings are empty. Please check the embedding process.")
        return None
    
    # Debugging: Print the embedding shape
    print(f"Embedding shape: {len(embedded_texts)} documents, {len(embedded_texts[0])} dimensions per document")
    
    # Create the FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    return vector_store

def detect_intent_and_extract_id(user_input, conversation_history):
    model = get_openai_model(temperature=0)
    
    response_schemas = [
        ResponseSchema(name="intent", description="The intent of the user's input: either 'document' or 'database'"),
        ResponseSchema(name="customer_id", description="Customer id in int format. If customer ID provided in the user input, set intent to 'database'.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    conversation_context = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in conversation_history[-6:]])

    prompt = f"""
    Given the following conversation history:
    {conversation_context}

    Analyze the following user input:
    "{user_input}"

    Determine the intent:
    {output_parser.get_format_instructions()}

    If no customer ID is provided, set "intent" to 'document' and 'customer_id' to 'null'.
    Take into account the entire conversation history when determining the intent and customer ID.
    """
    
    try:
        response = model.predict(prompt)
        return output_parser.parse(response)
    except Exception as e:
        print(f"Error parsing response: {e}")
        # Return a default response if parsing fails
        return {"intent": "unknown", "customer_id": None}


def get_conversational_chain():
    prompt_template = """
    Given the following conversation history and context, answer the question as detailed as possible. 
    If the answer is not in the provided context, just say, "I don't have enough information to answer that question." 
    Don't provide incorrect information. If it's a greeting respond nicely.

    Conversation History:
    {history}

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "history"])
    return load_qa_chain(get_openai_model(), chain_type="stuff", prompt=prompt)


def process_document_query(user_question, vector_store, conversation_history):
    try:
        docs = vector_store.similarity_search(user_question, k=10)
        chain = get_conversational_chain()
        history_text = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in conversation_history[-6:]])

        response = chain(
            {
                "input_documents": docs, 
                "question": user_question, 
                "history": history_text, 
                
            }, 
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        return f"Error processing query: {str(e)}"
    

def process_database_query(user_question, customer_id, conversation_history):
    customers = load_customer_data()
    for customer in customers['customers']:
        if customer_id == customer["customer_id"]:
            history_text = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in conversation_history[-6:]])
            
            prompt = f"""
            Given the following conversation history:
            {history_text}

            And the following customer information:
            {json.dumps(customer, indent=2)}

            Answer the user's question:
            {user_question}

            Provide a detailed and helpful response based on the customer data and conversation history.
            """
            return get_openai_model().predict(prompt)
    return f"Customer with ID {customer_id} not found in the database."


def load_response(func, var1, var2, var3, spinner_text):
    with st.spinner(spinner_text):
        response = func(var1, var2, var3)
    return response


def main():
    st.header("Document Genie Chat", divider="red")

    # File uploader for user to upload their own PDF
    uploaded_pdf = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_pdf is not None:
        st.session_state.vector_store = load_and_process_pdf(uploaded_pdf)
        st.success("PDF loaded and processed successfully!")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document or request database access"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            extracted_info = detect_intent_and_extract_id(prompt, st.session_state.conversation_history)
            
            if extracted_info['intent'] == "document":
                if st.session_state.vector_store:
                    response = load_response(process_document_query, prompt, st.session_state.vector_store, st.session_state.conversation_history, "Thinking...")
                else:
                    response = "Please upload a PDF first to process document queries."
            elif extracted_info['intent'] == "database":
                if extracted_info['customer_id'] and extracted_info['customer_id'] != "null":
                    response = load_response(process_database_query, prompt, int(extracted_info['customer_id']), st.session_state.conversation_history, "Processing database query...")
                else:
                    response = load_response(process_document_query, prompt, st.session_state.vector_store, st.session_state.conversation_history, "Thinking...")
            else:
                response = "An error occurred. Please try again."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.conversation_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()