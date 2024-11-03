# Contextual Chatbot Implementation in PyTorch

## Project Description

This project presents an advanced implementation of a contextual chatbot using PyTorch. The chatbot leverages deep learning techniques to understand and respond to user inputs in a conversational manner. The project is designed to be beginner-friendly while providing a comprehensive understanding of chatbot development. It includes a straightforward Feed Forward Neural Network with two hidden layers and allows for easy customization to fit various use cases.

### Technologies and Techniques Used

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Natural Language Processing**: NLTK (Natural Language Toolkit)
- **Model Architecture**: Feed Forward Neural Network
- **Training Data**: Customizable intents defined in `data.json`
- **Tokenization and Vocabulary**: Custom tokenizer and vocabulary files

## How to Run the Project

### Prerequisites

Ensure you have the following installed on your machine:
- Python 3.6 or higher
- `pip` (Python package installer)

### Installation


1. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

### Training the Model

1. **Customize Intents**
    Modify the `data.json` file to include your own patterns and responses.

2. **Run the Training Script**
    ```sh
    python train.py
    ```

### Using `tobin.py`

 1. **Run the Tobin Script**
        ```sh
        python tobin.py path to safetensors path to desired model.bin    note use same folder
        ```

2. **Functionality**
        The `tobin.py` script provides additional utilities for data preprocessing and analysis. It can be used to transform raw data into a format suitable for training and to perform exploratory data analysis.

### Running the Chatbot

1. **Start the Chatbot**
    ```sh
    python chat.py
    ```

2. **Interact with the Chatbot**
    The chatbot will prompt you to enter your queries and will respond based on the trained model.

### Additional Information

- **Hyperparameter Tuning**: You can adjust the hyperparameters in `hyperparameter_tuning.py` to optimize the model performance.
- **Model Checkpoints**: The trained model and tokenizer configurations are saved in the `fine_tuned_model_checkpoint` directory.
- **Fine-Tuning**: You can further fine-tune the model by modifying the training scripts and re-running the training process.
