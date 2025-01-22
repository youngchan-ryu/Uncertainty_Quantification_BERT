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

def evaluate_accuracy(test_dataloader, model_save_path, device):
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

    for batch in tqdm(test_dataloader):

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

    average_val_accuracy = val_accuracy / len(test_dataloader)
    average_val_loss = val_loss / len(test_dataloader)

    return average_val_accuracy, average_val_loss
### UQ
def enable_mc_dropout(model, dropout_rate):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = dropout_rate
            print(f"Enabling MC Dropout for {layer} - p={layer.p}")
            layer.train()

def disable_mc_dropout(model):
    model.eval()

def evaluate_uncertainty(test_dataloader, model_save_path, device, dropout_rate):
    # Conventional evaluating function
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model file not found at {model_save_path}")
    
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2)

    model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
    model.to(device)
    print(f"Model loaded from {model_save_path}")
    print("==========Evaluation started==========")
    
    ## UQ
    enable_mc_dropout(model, dropout_rate)
    num_samples = 10
    iteration_sample = 256
    if test_dataloader.batch_size != 1:
        raise ValueError("Batch size of test_dataloader must be 1 for uncertainty evaluation")
    # Create a new dataset where each element is the first element of the original test_dataloader
    # repeated_dataset = [[data[0], data[0], ... * iteration_sample], [data[1], data[1], ... * iteration_sample], ...]
    repeated_dataset = []
    for i in range(num_samples):
        repeated_dataset.append([test_dataloader.dataset[i]] * iteration_sample)

    confidence_list = []
    accuracy_list = []
    
    for i, sample_dataset in tqdm(enumerate(repeated_dataset)):
        sample_dataloader = torch.utils.data.DataLoader(
            sample_dataset,
            batch_size=16
        )

        pred_list = []
        logits_list = []
        correct_sample = 0

        for sample in tqdm(sample_dataloader):
            token_ids = sample[0]
            attention_mask = sample[1]
            labels = sample[2]

            with torch.no_grad():
                (loss, logits) = model(
                    token_ids,
                    attention_mask = attention_mask,
                    labels = labels,
                    token_type_ids = None,
                    return_dict=False)

            pred_flat = torch.argmax(logits, axis=1).flatten()
            pred_list += pred_flat.tolist()
            logits_list += logits.tolist()
            correct_sample += torch.sum(pred_flat == labels).item()


        pred_tensor = torch.tensor(pred_list)
        logits_tensor = torch.tensor(logits_list)
        probabilities = torch.nn.functional.softmax(torch.tensor(logits_list), dim=1)
        # prob[1] if pred is 1, prob[0] if pred is 0
        probabilities_tensor = torch.tensor([prob[1] if pred == 1 else prob[0] for pred, prob in zip(pred_list, probabilities)])

        # Calculate uncertainty
        mean = torch.mean(probabilities_tensor, axis=0)
        variance = torch.var(probabilities_tensor, axis=0)
        accuracy = correct_sample / len(pred_list)
        ci_95 = 1.96 * torch.sqrt(variance / len(pred_list))

        # print(f"pred_list : {pred_list}")
        # print(f"logits_list : {logits_list}")
        # print(f"probabilities : {probabilities.cpu().numpy()}")
        # print(f"probabilities_tensor : {probabilities_tensor.cpu().numpy()}")
        confidence_level = 0.95
        lower_percentile = (1 - confidence_level) / 2 * 100  # 2.5% for 95% CI
        upper_percentile = (1 + confidence_level) / 2 * 100  # 97.5% for 95% CI

        # Compute confidence intervals for each class
        confidence_intervals = np.percentile(probabilities.cpu().numpy(), [lower_percentile, upper_percentile], axis=0)

        print(f"\nStatistics for sample {i+1}")
        print(f"Mean: {mean}")
        print(f"Variance: {variance}")
        print(f"Accuracy: {accuracy}")
        print(f"95% Confidence Interval: ({mean - ci_95}, {mean + ci_95})")
        # Print the results
        for i in range(probabilities.shape[1]):
            print(f"Class {i}: 95% CI = [{confidence_intervals[0, i]}, {confidence_intervals[1, i]}]")

        confidence_list.append(mean)
        accuracy_list.append(accuracy)

    # Calculate ECE and MCE
    confidence_list = torch.tensor(confidence_list)
    accuracy_list = torch.tensor(accuracy_list)
    ece = torch.mean(torch.abs(confidence_list - accuracy_list))
    mce = torch.max(torch.abs(confidence_list - accuracy_list))

    print("==========All samples evaluated==========")
    print(f"\nExpected Calibration Error: {ece}")
    print(f"Maximum Calibration Error: {mce}")

    disable_mc_dropout(model)