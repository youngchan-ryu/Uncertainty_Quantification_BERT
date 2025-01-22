import os

from tqdm import tqdm
import torch
import numpy as np
from transformers import BertForSequenceClassification

def calculate_accuracy(preds, labels):
    """ Calculate the accuracy of model predictions against true labels.

    Parameters:
        preds (np.array): The predicted label from the model
        labels (np.array): The true label

    Returns:
        accuracy (float): The accuracy as a percentage of the correct
            predictions.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

    return accuracy

def calculate_accuracy_gpu(preds, labels):
    pred_flat = torch.argmax(preds, dim=1).flatten()
    labels_flat = labels.flatten()
    accuracy = torch.sum(pred_flat == labels_flat).item() / len(labels_flat)
    return accuracy

def evaluate_accuracy(val_dataloader, model_save_path, device):
    # Conventional evaluating function
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model file not found at {model_save_path}")
    
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2)

    model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
    model.to(device)
    print(f"Model loaded from {model_save_path}")
    model.eval()
    val_loss = 0
    val_accuracy = 0

    for batch in tqdm(val_dataloader):

        batch_token_ids = batch[0]
        batch_attention_mask = batch[1]
        batch_labels = batch[2]

        with torch.no_grad():
            (loss, logits) = model(
                batch_token_ids,
                attention_mask = batch_attention_mask,
                labels = batch_labels,
                token_type_ids = None,
                return_dict=False)

        # For CPU function
        # logits = logits.detach().cpu().numpy()
        # label_ids = batch_labels.to('cpu').numpy()
        # val_loss += loss.item()
        # val_accuracy += calculate_accuracy(logits, label_ids)

        val_loss += loss.item()
        val_accuracy += calculate_accuracy_gpu(logits, batch_labels)

    average_val_accuracy = val_accuracy / len(val_dataloader)
    average_val_loss = val_loss / len(val_dataloader)

    return average_val_accuracy, average_val_loss