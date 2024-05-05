import pandas as pd
import re
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Load the tokenizer and model
tokenizer = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-wtq')
model = TapasForQuestionAnswering.from_pretrained('google/tapas-large-finetuned-wtq')

# Function to clean and prepare text data
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)  # Convert to string if not already
    text = text.strip()  # Remove leading/trailing whitespace
    return re.sub(r"\s+", " ", text)  # Condense any multiple whitespace

# Load data from a CSV file with error handling for common issues
def load_and_clean_data(file_path):
    try:
        # Load the data
        data = pd.read_csv(file_path)
        # Clean and prepare string data for each column assumed to be text
        for column in data.columns:
            data[column] = data[column].apply(clean_text)
        return data
    except pd.errors.ParserError as e:
        print(f"Failed to parse CSV: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to process queries using TAPAS model
def answer_query(query, table):
    if table is not None:
        inputs = tokenizer(table=table, queries=[query], padding='max_length', return_tensors="pt")
        outputs = model(**inputs)
        predicted_answer_coordinates = tokenizer.convert_logits_to_predictions(
            inputs,
            outputs.logits.detach()
        )[0]

        # Decode the predicted answer coordinates
        answer = ""
        for coordinates in predicted_answer_coordinates:
            for coordinate in coordinates:
                row_index, column_index = coordinate
                if row_index < len(table) and column_index < len(table.columns):
                    answer += str(table.iat[row_index, column_index]) + " "

        return answer.strip()
    else:
        return "No data available for querying."

# Main execution block
if __name__ == "__main__":
    file_path = '../dataset/SCCM_Data.txt'  # Adjust this to your file path
    data = load_and_clean_data(file_path)

    # Example query
    query = "Give me devices in Sydney?"
    result = answer_query(query, data)
    print("Answer:", result)
