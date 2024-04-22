import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Data Load
def load_data(data_path):
    return pd.read_csv(data_path)

# Step 2: Pre-process Data
def preprocess_data(data):
    # Perform any necessary data preprocessing steps
    # This could include cleaning, tokenization, normalization, etc.
    if isinstance(data, str):
        return data.lower().strip()
    elif isinstance(data, int):
        return str(data)
    elif isinstance(data, float):
        return str(int(data))  # Convert float to int and then to string
    else:
        raise Exception("Unable to preprocess")

# Step 3: Feeding Data to a Pre-trained Model
def encode_data(data):
    # Use SentenceTransformer to encode the data into embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = data.iloc[:, 1].tolist()  # Assuming the text data is in the second column (index 1)
    embeddings = embedding_model.encode(texts)
    return embeddings

# Step 4: Storing Data to a Vector Store
def store_embeddings(embeddings, embeddings_path):
    # Store the embeddings to a vector store (e.g., CSV file)
    np.savetxt(embeddings_path, embeddings, delimiter=",")

# Step 5: Query Processing
def process_query(query, embeddings, data):
    # Use cosine similarity to find the most similar embedding to the query
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    most_similar_index = np.argmax(similarities)

    # Retrieve relevant information from the dataset based on the most similar index
    relevant_data = data.iloc[most_similar_index]
    return relevant_data

# Step 6: Response Sent Back to User
def generate_response(relevant_data):
    # Generate a response based on the relevant data retrieved from the dataset
    # This could involve formatting the data into a human-readable format.
    return relevant_data

# Main Function
if __name__ == "__main__":
    # Step 1: Data Load
    data_path = "../dataset/product-data.csv"
    data = load_data(data_path)

    # Step 2: Pre-process Data
    preprocessed_data=data.map(lambda x: preprocess_data(x) if isinstance(x, str) else x)
    #preprocessed_data = preprocess_data(data)

    # Step 3: Feeding Data to a Pre-trained Model
    embeddings = encode_data(preprocessed_data)

    # Step 4: Storing Data to a Vector Store
    embeddings_path = "../dataset/product-embeddings-gemini.csv"
    store_embeddings(embeddings, embeddings_path)

    # User Interaction
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        # Step 5: Query Processing
        relevant_data = process_query(user_query, embeddings, data)

        # Step 6: Response Sent Back to User
        response = generate_response(relevant_data)
        print(response)
