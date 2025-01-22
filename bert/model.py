from transformers import BertForSequenceClassification
from torch.optim import AdamW
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

def init_model(device):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2)

    model = model.to(device)

    return model

def init_optimizer(model):
    return AdamW(model.parameters(), lr=1e-5)

def init_loss_function():
    return nn.CrossEntropyLoss()

def init_scheduler(optimizer, train_dataloader, epochs):
    num_training_steps = epochs * len(train_dataloader)
    return get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps)