import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# Experiment with Different Embedding Models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Original model
# Try other models:
embedding_model = SentenceTransformer("all-mpnet-base-v2")
# Or:
# embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# Step 1: Data Storage
def load_data(data_path):
    data = pd.read_csv(data_path, na_values='NULL')
    return data

# Step 2: Data Preprocessing (if needed)
def preprocess_text(text):
    # Implement your text preprocessing logic here (e.g., punctuation removal, lemmatization)
    if isinstance(text, str):
        # Add your preprocessing steps here
        return text.lower().strip()
    else:
        return ''

# Step 3: Model Training
def train_embedding_model(data):
    # Separate embeddings for description and potentially name
    description_embeddings = embedding_model.encode(data['description'].apply(preprocess_text).tolist())
    # Optionally: name_embeddings = embedding_model.encode(data['product_name'].apply(preprocess_text).tolist())

    # Combine embeddings (e.g., concatenate or weighted average)
    embeddings = description_embeddings  # Replace with your combined embedding strategy

    return embeddings

# Step 4: Embedding Storage
def save_embeddings(embeddings, data_path):
    # Save embeddings to a structured format like CSV or Parquet
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.to_csv(data_path, index=False)

# Step 5: Query Processing (with additional checks and options)
def process_query(query, embeddings, data, k=3):
    # Preprocess query
    query = preprocess_text(query)
    query_embedding = embedding_model.encode([query])[0]

    # Search in the FAISS index (experiment with different index types)
    # index = faiss.IndexFlatL2(embeddings.shape[1])  # Original index
    quantizer = faiss.IndexFlatL2(embeddings.shape[1])  # Quantizer for the index
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist=100)
    index.train(embeddings)  # Train the quantizer
    index.add(embeddings)    # Add embeddings to the index
    distances, indices = index.search(query_embedding.reshape(1, -1), k=k)

    # Print shapes for debugging (remove later)
    print("Distances shape:", distances.shape)
    print("Indices shape:", indices.shape)

    # Retrieve relevant information
    results = []
    if indices:  # Check if indices is not empty
        for i in range(min(len(indices[0]), k)):  # Loop through valid indices up to k
            product_id = data.loc[indices[0][i], 'product_id']
            product_name = data.loc[indices[0][i], 'product_name']
            inventory = data.loc[indices[0][i], 'inventory']
            score = 1 - distances[0][i]
            results.append((product_id, product_name, inventory, score))

    # Sort results by score (highest similarity first) and consider inventory
    results.sort(key=lambda x: (x[3], x[2] == "In Stock"), reverse=True)
    return results

# Response Generation
def generate_response(results):
    if results:
        product_id, product_name, inventory, score = results[0]
        response = f"We have the {product_name} (ID: {product_id}). Availability: {inventory}"
    else:
        response = "No matching products found. Please try a different query."
    return response

# Main Function
if __name__ == "__main__":
    # Step 1: Data Storage
    data_path = "../dataset/product-data.csv"  # Replace with your actual data path
    data = load_data(data_path)

    # Step 2: Data Preprocessing (if needed)
    data = data.map(lambda x: preprocess_text(x) if isinstance(x, str) else x)

    # Step 3: Model Training
    embeddings = train_embedding_model(data)

    # Step 4: Embedding Storage (optional - for future use)
    embeddings_path = "../dataset/product-embeddings2.csv"  # Replace with your desired path
    save_embeddings(embeddings, embeddings_path)

    # Step 5: Query Processing and User Interaction
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        results = process_query(user_query, embeddings, data)
        response = generate_response(results)
        print(response)