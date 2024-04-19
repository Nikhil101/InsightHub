import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Data Loading and Preprocessing
def load_data(data_path):
    data = pd.read_csv(data_path)
    data["description"] = data["description"].apply(preprocess_text)
    return data

def preprocess_text(text):
    # Implement your text preprocessing logic here
    return text.lower().strip()

# Embedding Generation
def generate_embeddings(data):

    embeddings = embedding_model.encode(data["description"])
    return embeddings

# Response Generation
def generate_response(query, data_with_embeddings):
    query_embedding = embedding_model.encode(query)
    similarities = cosine_similarity(query_embedding.reshape(1, -1), data_with_embeddings["embeddings"])
    most_similar_index = np.argmax(similarities)

    product_description = data_with_embeddings["description"][most_similar_index]
    inventory_status = data_with_embeddings["inventory"][most_similar_index]

    response = f"Based on your query, you might be interested in a product with the following description: '{product_description}'\n"
    response += f"Inventory status: {inventory_status}"
    return response

# Main Function
if __name__ == "__main__":
    data_path = "../dataset/product-data.csv"
    data = load_data(data_path)
    embeddings = generate_embeddings(data)
    data_with_embeddings = {"description": data["description"], "embeddings": embeddings, "inventory": data["inventory"]}

    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        response = generate_response(user_query, data_with_embeddings)
        print(response)