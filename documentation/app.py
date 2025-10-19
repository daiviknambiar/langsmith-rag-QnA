from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from flask_cors import CORS
from openai import embeddings


app = Flask(__name__)
CORS(app)

def initialize_rag():
    loader = DirectoryLoader('documentation/', glob="**/*.html", loader_cls=BSHTMLLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()            # requires OPENAI_API_KEY
    vectorstore = FAISS.from_documents(texts, embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    return qa_chain


#initialize rag
qa_chain = initialize_rag()

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = qa_chain.run(question)
    print(answer)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
