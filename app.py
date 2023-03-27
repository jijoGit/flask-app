from flask import Flask, request
from flask_cors import CORS
import torch
from transformers import BertTokenizer,  BertForSequenceClassification
import os
import pickle
from sklearn.preprocessing import LabelEncoder


os.environ['CURL_CA_BUNDLE'] = ''


# Load the saved model and optimizer state dict
model_path = 'model_state_dict.pt'
optimizer_path = 'optimizer_state_dict.pt'
num_labels = 13485
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=num_labels)
model.load_state_dict(torch.load(model_path))
optimizer_state_dict = torch.load(optimizer_path)
learning_rate = 2e-5

# Load the label encoder from file
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

app = Flask(__name__)
CORS(app)  # enable CORS for all routes


@app.route("/")
def hello():
    return "Hello, World!"

# Define the route for API


@app.route('/classify', methods=['POST'])
def classify():
    # Get the input text from the request
    text = request.json['text']

    # # Preprocess the input text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Set the maximum sequence length
    max_seq_length = 128

    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_seq_length-2]
    # Add the [CLS] and [SEP] tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # Convert the tokens to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Create attention masks
    attention_mask = [1] * len(input_ids)
    # Pad the input sequence
    padding_length = max_seq_length - len(input_ids)
    input_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    # Convert the input IDs and attention masks to tensors
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)

    # inputs = tokenizer(text, return_tensors='pt',
    #                    padding=True, truncation=True)
    # input_ids = inputs['input_ids']
    # attention_mask = inputs['attention_mask']

    # # Classify the input text
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = label_encoder.inverse_transform(
            [logits.argmax().item()])[0]
        print(predicted_label)
        return {'predicted_label': predicted_label}
    # Decode predicted labels using the label encoder

    # Return the predicted label
    return {'predicted_label': False}


if __name__ == "__main__":
    app.run()


# curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"Assign and manag\"}" http://localhost:5000/classify
