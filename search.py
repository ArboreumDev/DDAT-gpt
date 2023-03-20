import requests
import openai
import pandas as pd
from haystack.document_store import MemoryDocumentStore
from haystack.pipeline import ExtractiveQAPipeline
from haystack.retriever.dense import EmbeddingRetriever
from farm.modeling.tokenization import Tokenizer
from farm.modeling.language_model import LanguageModel
from farm.data_handler.processor import TextSimilarityProcessor
from farm.infer import Inferencer
from flask import Flask, request, jsonify


# Configure OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to fetch document from Worldbank API
def fetch_document(p_number):
    url = f"http://api.worldbank.org/v2/projects/{p_number}?format=json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        project = data[1][0]
        title = project["project_name"]
        description = project["project_abstract"]["cdata"]

        return {
            "text": title + "\n" + description,
            "meta": {
                "p_number": p_number,
            },
        }
    else:
        return None


# Function to get document embeddings using OpenAI API
def get_embeddings(texts):
    response = openai.Embed.create(documents=texts)
    return response["data"]


# Function to initialize the semantic search pipeline
def init_search_pipeline():
    # Initialize document store
    document_store = MemoryDocumentStore()

    # Initialize tokenizer and model
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="bert-base-uncased",
        do_lower_case=True,
    )
    language_model = LanguageModel.load("bert-base-uncased")
    processor = TextSimilarityProcessor(tokenizer=tokenizer)
    model = Inferencer(processor=processor, language_model=language_model)

    # Initialize EmbeddingRetriever
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=model,
        use_gpu=True,
        model_format="farm",
    )

    # Initialize the search pipeline
    search_pipeline = ExtractiveQAPipeline(retriever)

    return search_pipeline, document_store


# Initialize Flask app
app = Flask(__name__)
search_pipeline, document_store = init_search_pipeline()


@app.route("/add_document", methods=["POST"])
def add_document():
    p_number = request.form["p_number"]
    document = fetch_document(p_number)

    if document:
        embeddings = get_embeddings([document["text"]])[0]["embeddings"]
        document["embedding"] = embeddings
        document_store.write_documents([document])

        return jsonify({"status": "success", "p_number": p_number}), 200
    else:
        return jsonify({"status": "failure", "message": "Failed to fetch document."}), 400


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")

    if not query:
        return jsonify({"status": "failure", "message": "No query provided."}), 400

        results = search_pipeline.run(query=query, top_k_retriever=5)
        
    matched_documents = [
        {
            "p_number": result["document"]["meta"]["p_number"],
            "score": result["score"],
            "text": result["document"]["text"],
        }
        for result in results
    ]

    return jsonify({"status": "success", "results": matched_documents}), 200


if __name__ == "__main__":
    app.run(debug=True)
