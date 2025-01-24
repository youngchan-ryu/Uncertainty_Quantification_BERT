import os

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
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

def load_model(model, model_save_path):
    state_dict = torch.load(model_save_path)['model_state_dict']
    
    # Remove the 'module.' prefix from keys
    # Mismatch from DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # Load the modified state dict
    model.load_state_dict(new_state_dict)

def evaluate_accuracy(test_dataloader, model_save_path, device):
    # Conventional evaluating function
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model file not found at {model_save_path}")
    
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2)

    load_model(model, model_save_path)
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

def reliability_diagram(confidence_list: list, is_correct_list: list):
    num_bins = 10
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)  # Bin edges from 0 to 1
    bin_indices = np.digitize(confidence_list, bin_edges, right=True)
    bin_middlepoint = (bin_edges[1:] + bin_edges[:-1])/2

    bin_confidences = []
    bin_accuracies = []
    bin_gaps = []
    confidence_nparray = np.array(confidence_list)
    is_correct_nparray = np.array(is_correct_list)
    # ECE is weighted average of calibration error in each bin
    # MCE is maximum calibration error in each bin
    cum_ce = 0
    mce = 0

    # organizing bin elements
    for i in range(1, num_bins + 1):
        indices = np.where(bin_indices == i)[0]  # Get indices of elements in the bin
        if len(indices) > 0:
            avg_confidence = np.mean(confidence_nparray[indices])  # Average confidence
            avg_accuracy = np.mean(is_correct_nparray[indices])  # Accuracy as mean of correct labels
            gap = avg_confidence - avg_accuracy  # Gap between confidence and accuracy

            bin_confidences.append(avg_confidence)
            bin_accuracies.append(avg_accuracy)
            bin_gaps.append(gap)
            cum_ce += np.abs(gap) * len(indices)
            mce = max(mce, np.abs(gap))
        else:
            bin_confidences.append(0)
            bin_accuracies.append(0)
            bin_gaps.append(0)
    
    ece = cum_ce / len(confidence_list)

    # ECE/MCE statistics
    print("==========All samples evaluated==========")
    print(f"\nExpected Calibration Error: {ece}")
    print(f"Maximum Calibration Error: {mce}")

    # drawing plot
    bar_width = 0.08  # Width of the bars
    plt.figure(figsize=(8, 6))

    plt.bar(bin_edges[:-1], bin_accuracies, width=bar_width, align='edge', color='blue', edgecolor='black', label="Outputs")
    plt.bar(bin_edges[:-1], bin_gaps, width=bar_width, align='edge', color='pink', alpha=0.7, label="Gap", bottom=bin_accuracies)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")

    plt.text(0.7, 0.1, f'ECE={ece:.4f}', fontsize=14, bbox=dict(facecolor='lightgray', alpha=0.5))

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('test_result/reliability_diagram_box.png', dpi=300, bbox_inches='tight')


def uncertainty_statistics_one_sample(pred_list: list, probabilities_list: list, probability_sample: list, sample_prediction: int, label: int, is_correct: int):
    # prob[1] if pred is 1, prob[0] if pred is 0
    sample_pred_probabilities_list = [prob[sample_prediction] for prob in probabilities_list]
    sample_pred_prob = probability_sample[sample_prediction] 

    # Calculate uncertainty
    mean = torch.mean(torch.tensor(sample_pred_probabilities_list), axis=0)
    variance = torch.var(torch.tensor(sample_pred_probabilities_list), axis=0)
    ci_95 = 1.96 * torch.sqrt(variance / len(pred_list))

    confidence_level = 0.95
    lower_percentile = (1 - confidence_level) / 2 * 100  # 2.5% for 95% CI
    upper_percentile = (1 + confidence_level) / 2 * 100  # 97.5% for 95% CI

    # Compute confidence intervals for each class
    confidence_intervals = np.percentile(probabilities_list, [lower_percentile, upper_percentile], axis=0)

    print(f"\nStatistics for sample")
    print(f"Final prediction: {sample_prediction}")
    print(f"True label: {label}")
    print(f"Correct: {True if is_correct else False}")
    print(f"Prediction probability: {sample_pred_prob}")
    print(f"Each stochastic passes..")
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"95% Confidence Interval: ({mean - ci_95}, {mean + ci_95})")
    # Print the results
    for i in range(2):
        print(f"Class {i}: 95% CI = [{confidence_intervals[0, i]}, {confidence_intervals[1, i]}]")

def evaluate_uncertainty(test_dataloader, model_save_path, device, dropout_rate):
    # Conventional evaluating function
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model file not found at {model_save_path}")
    
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2)

    load_model(model, model_save_path)
    model.to(device)
    print(f"Model loaded from {model_save_path}")
    print("==========Evaluation started==========")
    
    ## UQ
    enable_mc_dropout(model, dropout_rate)
    # Settings
    num_samples = 32
    iteration_sample = 256
    threshold = 0.5

    if test_dataloader.batch_size != 1:
        raise ValueError("Batch size of test_dataloader must be 1 for uncertainty evaluation")
    # Create a new dataset where each element is the first element of the original test_dataloader
    # repeated_dataset = [[data[0], data[0], ... * iteration_sample], [data[1], data[1], ... * iteration_sample], ...]
    repeated_dataset = []
    for i in range(num_samples):
        repeated_dataset.append([test_dataloader.dataset[i]] * iteration_sample)

    confidence_list = []
    is_correct_list = []
    
    for i, sample_dataset in tqdm(enumerate(repeated_dataset)):
        sample_dataloader = torch.utils.data.DataLoader(
            sample_dataset,
            batch_size=16
        )

        pred_list = [] # [iteration_sample] : individual prediction within one sample is not important
        probabilities_list = [] # [iteration_sample, 2] 
        label_counter = 0 # counter checking for all sample label are same

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
            probabilities_list += torch.nn.functional.softmax(logits, dim=1).tolist() 
            label_counter += torch.sum(labels)

        # Check if sum of labels are iteration_sample (which means all labels are 1) or 0 (which means all labels are 0)
        if label_counter != 0 and label_counter != iteration_sample:
            raise ValueError("All labels in one sample must be same")
        else:
            label = 1 if label_counter == iteration_sample else 0
        
        probability_sample = torch.mean(torch.tensor(probabilities_list), dim=0) # [2]
        sample_prediction = 1 if probability_sample[1] > threshold else 0
        is_correct = 1 if sample_prediction == label else 0

        confidence = probability_sample[sample_prediction]
        confidence_list.append(confidence)
        is_correct_list.append(is_correct)

        uncertainty_statistics_one_sample(pred_list, probabilities_list, probability_sample, sample_prediction, label, is_correct)

    reliability_diagram(confidence_list, is_correct_list)

    disable_mc_dropout(model)