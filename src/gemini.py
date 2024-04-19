import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Step 1: Data Storage
def load_data(data_path):
    data = pd.read_csv(data_path,na_values='NULL')
    return data

# Step 2: Data Preprocessing (if needed)
def preprocess_text(text):
    # Implement your text preprocessing logic here
    if isinstance(text, str):
        return text.lower().strip()
    elif isinstance(text, int):
        return str(text)
    elif isinstance(text, float):
        return str(int(text))  # Convert float to int and then to string
    else:
        return ''

# Step 3: Model Training
def train_embedding_model(data):

    embeddings = embedding_model.encode(data.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist())  # Combining all columns into one string
    return embeddings

# Step 4: Embedding Storage
def save_embeddings(embeddings, data_path):
    # Save embeddings to a structured format like CSV or Parquet
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.to_csv(data_path, index=False)

# Step 5: Query Processing
def process_query(query, vectorizer, embeddings, data):
    # Preprocess query
    query_vector = vectorizer.transform([query])

    # Calculate similarity between query vector and stored embeddings
    similarities = cosine_similarity(query_vector, embeddings)
    most_similar_index = similarities.argmax()

    # Retrieve relevant information
    response_data = data.iloc[most_similar_index]
    return response_data

# # Response Generation
# def generate_response(query, data_with_embeddings):
#     query_embedding = embedding_model.encode(query)
#     similarities = cosine_similarity(query_embedding.reshape(1, -1), data_with_embeddings["embeddings"])
#     most_similar_index = np.argmax(similarities)
#
#     product_description = data_with_embeddings["description"][most_similar_index]
#     inventory_status = data_with_embeddings["inventory"][most_similar_index]
#
#     response = f"Based on your query, you might be interested in a product with the following description: '{product_description}'\n"
#     response += f"Inventory status: {inventory_status}"
#     return response

# Main Function
if __name__ == "__main__":
    # Step 1: Data Storage
    data_path = "../dataset/product-data.csv"  # Adjust the path accordingly
    data = load_data(data_path)

    # Step 2: Data Preprocessing (if needed)
    data=data.map(lambda x: preprocess_text(x) if isinstance(x, str) else x)
    print(data)

    # Step 3: Model Training
    embeddings = train_embedding_model(data)

    # Step 4: Embedding Storage
    embeddings_path = "../dataset/product-embeddings.csv"
    save_embeddings(embeddings, embeddings_path)

    # Step 5: Query Processing
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data.apply(lambda x: ' '.join(x.astype(str)), axis=1))
    embeddings = pd.read_csv(embeddings_path)

    # Ensure query vector and embeddings have the same number of features
    embeddings = embeddings.to_numpy()[:, :vectorizer.transform(['']).shape[1]]

    #data_with_embeddings = {"description": data["description"], "embeddings": embeddings, "inventory": data["inventory"]}

    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        response = process_query(user_query, vectorizer, embeddings, data)
        print(response)