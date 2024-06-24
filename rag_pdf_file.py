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


current_pdf_file = None
current_vectorstore = None
current_retriever = None

def process_input(pdf_file, question, useless):
    global current_pdf_file, current_vectorstore, current_retriever
    model_local = ChatOllama(model="mistral")
    #print(f"pdf_file: {pdf_file}, question: {question}, useless: {useless}")

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
        #print(doc_splits)

        current_vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
        )
        #print(vectorstore)
        current_retriever = current_vectorstore.as_retriever()
        #print(vectorstore.vectors)

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": current_retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)


def update_textbox(file_data):
    #print(f"Uploaded file: {file_data.name}")
    print(f"Uploaded file data: {file_data}")
    return file_data



with gr.Blocks() as iface:
    uploadButton = gr.UploadButton("Click to Upload a File", file_types=[".pdf"])
    textBoxUploadedFile = gr.Textbox(label="PDF File Name", interactive=False)
    questionBox = gr.Textbox(label="Question")

    uploadButton.upload(update_textbox, uploadButton, textBoxUploadedFile)
    
    #iface.add([uploadButton, questionBox, textBoxUploadedFile])
    gr.Interface(
        fn=process_input,
        inputs=[uploadButton, questionBox, textBoxUploadedFile],
        outputs="text",
        title="Document Query with Ollama",
        description="Upload a PDF file and ask a question to query the document."
    )
iface.launch()
