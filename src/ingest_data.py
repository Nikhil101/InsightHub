import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import sklearn

df = pd.read_csv('../dataset/SCCM-Sample-DataSet.csv',na_values='NULL')
print(df.head())
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(df).toList()

# Split the data into training and validation sets
train_size = int(0.8 * len(df))
train_tokens = tokens[:train_size]
val_tokens = tokens[train_size:]

# Create a synthetic dataset for the validation set
synthetic_data = pd.DataFrame({'text': val_tokens})

# Define the LLM model and optimizer
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
optimizer = AdamW(learning_rate=1e-5, weight_decay=0.01)

# Train the LLM model on the training data
train_data = train_test_split(train_tokens, labels=df['label'], test_size=0.2, random_state=42)
model.train(train_data, optimizer=optimizer)

# Evaluate the LLM model on the validation data
val_preds = model(synthetic_data['text'])
f1 = f1_score(df['label'], val_preds, average='macro')
print(f'Validation F1 score: {f1}')

# Save the trained LLM model
model.save_pretrained('bert-base-uncased')