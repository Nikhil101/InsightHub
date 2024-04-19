import pandas as pd

def ingest_data(file_path):
    """
    Ingest data from a CSV file.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - DataFrame: Pandas DataFrame containing the ingested data.
    """
    # Read CSV file into DataFrame
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    # Path to the CSV file
    csv_file = "../dataset/SCCM-Sample-DataSet.csv"

    # Ingest data
    data_df = ingest_data(csv_file)

    # Display the ingested data
    print("Ingested Data:")
    print(data_df)
