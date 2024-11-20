# Contextual Chatbot Implementation in PyTorch

## Project Description

This project presents an advanced implementation of a contextual chatbot using PyTorch. The chatbot leverages deep learning techniques to understand and respond to user inputs in a conversational manner. The project is designed to be beginner-friendly while providing a comprehensive understanding of chatbot development. It includes a straightforward Feed Forward Neural Network with two hidden layers and allows for easy customization to fit various use cases.

## Sample Outputs

```text
Question: Can you help me with something?
Response: Sure. Tell me how can i assist you
--------------------------------------------------
Question: Can you help me with something?
Response: Tell me your problem so that i can assist you
--------------------------------------------------
Question: Can you help me with something?
Response: Yes, sure. How can I help you?
--------------------------------------------------
Question: I am feeling lonely
Response: Why do you think you feel this way?
--------------------------------------------------
Question: I feel so worthless.
Response: I'm trying my best to help you. So please talk to me
--------------------------------------------------
Question: I feel so worthless.
Response: I'm sorry to hear that. I'm doing my best to help
--------------------------------------------------
Question: I feel so anxius.
Response: Did something happen which made you feel this way?
--------------------------------------------------
Question: I can't take it anymore
Response: I'm trying my best to help you. So please talk to me
--------------------------------------------------
Question: I am so stressed out
Response: Give yourself a break. Go easy on yourself.
--------------------------------------------------
Question: I am so stressed out
Response: What do you think is causing this?
--------------------------------------------------
Question: I am so stressed out
Response: Give yourself a break. Go easy on yourself.
--------------------------------------------------
Question: I have insominia
Response: That seem awful. What do you think is behind this?
--------------------------------------------------
Question: I'm scared
Response: That's geat to hear. I'm glad you're feeling this way.
--------------------------------------------------
Question: I'm scared
Response: Oh i see. That's great.
--------------------------------------------------
Question: I'm scared
Response: That's geat to hear. I'm glad you're feeling this way.
--------------------------------------------------
Question: My mom died
Response: I'm sorry to hear that. If you want to talk about it. I'm here.
--------------------------------------------------
Question: My mom died
Response: I am really sorry to hear that. I am here to help you with grief, anxiety and anything else you may feel at this time.
--------------------------------------------------
Question: My mom died
Response: My condolences. I'm here if you need to talk.
--------------------------------------------------
Question: You don't understand me.
Response: I'm sorry to hear that. I'm doing my best to help
--------------------------------------------------
Question: You don't understand me.
Response: I'm sorry to hear that. I'm doing my best to help
--------------------------------------------------
Question: You don't understand me.
Response: It sound like i'm not being very helpful right now.
--------------------------------------------------
Question: That's all.
Response: I'm listening. Please go on.
--------------------------------------------------
Question: That's all.
Response: Come elucidate your thoughts
--------------------------------------------------
Question: That's all.
Response: Let's discuss further why you're feeling this way.
--------------------------------------------------
```
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
    python tobin.py path_to_safetensors path_to_desired model.bin
    ```  
    note use same folder(Fine Tuned Model)
3. **Functionality**
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

- ## Contact

- **GitHub**: Rayyan9477 (https://github.com/Rayyan9477)
- **LinkedIn**: Rayyan Ahmed (https://www.linkedin.com/in/rayyan-ahmed9477/)
- **Email**: rayyanahmed265@yahoo.com

