from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from huggingface_hub import login
import os

login("your_token")

# Step 1: Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Assign a padding token if not set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Step 2: Load and preprocess dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations")

# Split the dataset into train and validation sets
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]

# Tokenization function with labels
def tokenize_function(examples):
    # Combine 'Context' and 'Response' for training
    inputs = [f"Context: {context}\nResponse: {response}" for context, response in zip(examples['Context'], examples['Response'])]
    # Tokenize and prepare labels
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    model_inputs["labels"] = model_inputs["input_ids"].copy()  # Set labels as input_ids for causal language modeling
    return model_inputs

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)

# Set dataset format for PyTorch
tokenized_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Step 3: Define training arguments
training_args = TrainingArguments(
    output_dir="./mental_health_chatbot_model",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    push_to_hub=False,
    fp16=True if torch.cuda.is_available() else False,
)

# Step 4: Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
)

# Step 5: Fine-tune the model
trainer.train()

# Step 6: Save the fine-tuned model for local use
trainer.save_model("./mental_health_chatbot_model")
tokenizer.save_pretrained("./mental_health_chatbot_model")

print("Fine-tuning complete. Model saved locally.")