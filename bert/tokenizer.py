import torch
import pickle
from transformers import BertTokenizer


def tokenize_data(df):
    token_ids = []
    attention_masks = []
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for i, review in enumerate(df['review_cleaned']):
        if i % 100 == 0:
            print(f"{i} data processed")
        
        batch_encoder = tokenizer.encode_plus(
            review,
            max_length = 512,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )

        token_ids.append(batch_encoder['input_ids'])
        attention_masks.append(batch_encoder['attention_mask'])

    token_ids = torch.cat(token_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    labels = torch.tensor(df['sentiment_encoded'].values)

    return token_ids, attention_masks, labels

def save_tokenized_data(token_ids, attention_masks, labels, output_file):
    data = {'token_ids' : token_ids, 'attention_masks' : attention_masks, 'labels' : labels}

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Tokenized data saved as {output_file}")

def load_tokenized_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Tokenized data loaded from {file_path}")
    return data['token_ids'], data['attention_masks'], data['labels']