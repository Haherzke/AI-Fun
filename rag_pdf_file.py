import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import json
import os

current_pdf_file = None
current_vectorstore = None
current_retriever = None
CHAT_HISTORY_FILE = "chat_history.json"

# Define templates for both languages
# Define enhanced templates for both languages
templates = {
    "English": """You are an intelligent assistant capable of analyzing documents and engaging in human-like conversation. Your primary focus is to provide insights and answers based on the given context. However, you can also engage in general conversation if the user's question is not directly related to the document.

Given context: {context}

User question: {question}

Instructions:
1. If the question is related to the context, provide a detailed analysis and answer based solely on the information given.
2. If the question is not directly related to the context but is a general query, respond in a friendly, conversational manner.
3. If you're unsure whether the question relates to the context, you may ask for clarification.
4. Always maintain a polite and helpful tone.

Please provide your response:
    """,
    "Deutsch": """Sie sind ein intelligenter Assistent, der Dokumente analysieren und menschenähnliche Gespräche führen kann. Ihr Hauptfokus liegt darauf, Erkenntnisse und Antworten basierend auf dem gegebenen Kontext zu liefern. Sie können jedoch auch an allgemeinen Gesprächen teilnehmen, wenn die Frage des Benutzers nicht direkt mit dem Dokument zusammenhängt.

Gegebener Kontext: {context}

Benutzerfrage: {question}

Anweisungen:
1. Wenn sich die Frage auf den Kontext bezieht, liefern Sie eine detaillierte Analyse und Antwort ausschließlich basierend auf den gegebenen Informationen.
2. Wenn die Frage nicht direkt mit dem Kontext zusammenhängt, sondern eine allgemeine Anfrage ist, antworten Sie freundlich und gesprächig.
3. Wenn Sie sich nicht sicher sind, ob sich die Frage auf den Kontext bezieht, können Sie um Klärung bitten.
4. Behalten Sie immer einen höflichen und hilfsbereiten Ton bei.

Bitte geben Sie Ihre Antwort:
    """
}

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def process_input(pdf_file, question, useless, language):
    global current_pdf_file, current_vectorstore, current_retriever
    model_local = ChatOllama(model="mistral")
    # Define a path for storing the vector database
    persist_directory = "./chroma_db"

    if current_pdf_file is None or current_pdf_file != pdf_file:
        current_pdf_file = pdf_file 
        # Load the uploaded PDF file
        docs = [PyPDFLoader(pdf_file).load()]
        print(f"Loaded {len(docs)} documents from PDF file.")
        docs_list = [item for sublist in docs for item in sublist]
        print(f"Loaded {len(docs_list)} pages from PDF file.")
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs_list)
        print(f"Split {len(doc_splits)} documents into chunks.")

        current_vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
        )
        current_retriever = current_vectorstore.as_retriever()

    # Use the template for the selected language
    after_rag_template = templates[language]
    
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": current_retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    answer = after_rag_chain.invoke(question)
    return question, answer

def update_textbox(file_data):
    print(f"Uploaded file data: {file_data}")
    return file_data

def update_chat_history(history, file, question, useless, language):
    pdf_name = os.path.basename(file.name) if file else "No PDF uploaded"
    question, answer = process_input(file, question, useless, language)
    new_entry = (f"PDF: {pdf_name}\nQ: {question}", answer)
    history.append(new_entry)
    save_chat_history(history)
    return history, ""

with gr.Blocks() as iface:
    uploadButton = gr.UploadButton("Click to Upload a File", file_types=[".pdf"])
    textBoxUploadedFile = gr.Textbox(label="PDF File Name", interactive=False)
    uploadButton.upload(update_textbox, uploadButton, textBoxUploadedFile)

    language_selector = gr.Radio(["English", "Deutsch"], label="Language", value="English")

    chatbot = gr.Chatbot(label="Chat History", value=load_chat_history())
    with gr.Row():
        questionBox = gr.Textbox(label="Question", placeholder="Type your question here...", scale=4)
        submitBtn = gr.Button("Submit", scale=1)

    submitBtn.click(
        update_chat_history,
        inputs=[chatbot, uploadButton, questionBox, textBoxUploadedFile, language_selector],
        outputs=[chatbot, questionBox]
    )
    
    questionBox.submit(
        update_chat_history,
        inputs=[chatbot, uploadButton, questionBox, textBoxUploadedFile, language_selector],
        outputs=[chatbot, questionBox]
    )

iface.launch()