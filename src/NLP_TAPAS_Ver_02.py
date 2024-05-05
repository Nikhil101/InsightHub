import pandas as pd
import re
import logging
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the tokenizer and model
tokenizer = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-wtq')
model = TapasForQuestionAnswering.from_pretrained('google/tapas-large-finetuned-wtq')

def clean_text(text):
    """Clean and prepare text data."""
    if text is None:
        cleaned_text = ''
    else:
        cleaned_text = str(text).strip()
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text

def load_and_clean_data(file_path, column_types):
    """Load data from a CSV file and apply cleaning to text columns."""
    try:
        data = pd.read_csv(file_path, dtype=column_types)
        for column in data.columns:
            data[column] = data[column].apply(clean_text)
        return data
    except pd.errors.ParserError as e:
        logging.error(f"Failed to parse CSV: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

def process_query(query, table):
    """Process queries using the TAPAS model and return results including aggregations."""
    if table is not None:
        query = clean_text(query)
        inputs = tokenizer(table=table, queries=[query], padding='max_length', return_tensors="pt")
        outputs = model(**inputs)

        # Get answer coordinates and check for aggregation indices
        results = tokenizer.convert_logits_to_predictions(
            inputs,
            outputs.logits.detach()
        )
        predicted_answer_coordinates = results[0]  # Always present
        predicted_aggregation_indices = results[1].argmax(-1).tolist() if len(results) > 1 else None

        # Generate answer text from coordinates
        answer = ""
        if predicted_answer_coordinates:  # Ensure there are coordinates
            for coordinates in predicted_answer_coordinates:
                for row_index, column_index in coordinates:
                    if isinstance(row_index, int) and isinstance(column_index, int):
                        answer += str(table.iat[row_index, column_index]) + " "

        # Log the outputs
        logging.info(f"Predicted Answer Coordinates: {predicted_answer_coordinates}")
        logging.info(f"Predicted Aggregation Indices: {predicted_aggregation_indices}")
        logging.info(f"Answer: {answer.strip()}")

        return answer.strip(), predicted_answer_coordinates, predicted_aggregation_indices
    else:
        return "No data available for querying.", None, None

if __name__ == "__main__":
    file_path = '../dataset/SCCM_Data.txt'  # Adjust this to your file path
    column_types = {
        'Device Name': str, 'Operating System': str, 'Processor': str,
        'RAM': str, 'Storage': str, 'Location': str, 'Cost': int  # Ensure proper types are set
    }
    data = load_and_clean_data(file_path, column_types)

    # Example query
    query = "What is the total cost of laptops in New York?"
    answer, coordinates, aggregation_indices = process_query(query, data)
    logging.info(f"Final Answer: {answer}")
