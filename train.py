import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        encoding = self.tokenizer.encode_plus(
            item['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def train_model():
    if os.path.exists('fine_tuned_model'):
        print("Model already trained and saved.")
        return

    with open('data.json', 'r') as f:
        intents = json.load(f)

    data = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            data.append({'text': pattern, 'label': intents['intents'].index(intent)})

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = ChatDataset(data, tokenizer, max_len=128)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(intents['intents'])
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 5  # 5 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    def train(model, dataloader, optimizer, scheduler, device):
        model.train()
        for epoch in range(5):
            print(f"Epoch {epoch + 1}/5")
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item()}")

                if batch_idx % 100 == 0:
                    model.save_pretrained('fine_tuned_model_checkpoint')
                    tokenizer.save_pretrained('fine_tuned_model_checkpoint')

    try:
        train(model, train_loader, optimizer, scheduler, device)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        torch.cuda.empty_cache()

    model.save_pretrained('fine_tuned_model')
    tokenizer.save_pretrained('fine_tuned_model')

    required_files = [
        'config.json', 'pytorch_model.bin', 'tokenizer_config.json', 'vocab.txt'
    ]
    missing_files = [
        f for f in required_files if not os.path.exists(
            os.path.join('fine_tuned_model', f)
        )
    ]
    if missing_files:
        print(f"Warning: The following files are missing: {missing_files}")
    else:
        print("All necessary files are saved successfully.")

if __name__ == "__main__":
    train_model()