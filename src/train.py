from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(self.data[idx], add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors='pt')
        return inputs

def fine_tune_model(data, prompts):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    dataset = CustomDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained('fine_tuned_model')
    tokenizer.save_pretrained('fine_tuned_model')


if __name__ == "__main__":
    # Sample Data and Prompts
    sample_data = [
        "The quick brown fox jumps over the lazy dog.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "To be, or not to be, that is the question.",
        "I wandered lonely as a cloud that floats on high o'er vales and hills.",
        "The cat in the hat.",
        "All the world's a stage, and all the men and women merely players.",
        "To infinity and beyond!",
        "A penny saved is a penny earned."
    ]

    sample_prompts = [
        "Can you provide details about a specific sentence?",
        "Tell me about a sentence with a specific word.",
        "What are the characteristics of a sentence starting with a specific letter?",
        "Give me information about sentences containing a particular phrase.",
        "Which sentences have a certain length?",
        "List sentences with a specific structure.",
        "Provide details about sentences with a certain sentiment.",
        "Give me examples of sentences from a specific author."
    ]

    # Fine-tune model
    fine_tune_model(sample_data, sample_prompts)
