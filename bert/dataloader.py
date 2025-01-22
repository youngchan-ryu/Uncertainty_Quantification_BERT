import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def split_load_data(token_ids, attention_masks, labels, device, uncertainty_flag):
    val_size = 0.1

    # Split the token IDs
    train_ids, val_ids = train_test_split(
                            token_ids,
                            test_size=val_size,
                            shuffle=False)

    # Split the attention masks
    train_masks, val_masks = train_test_split(
                                attention_masks,
                                test_size=val_size,
                                shuffle=False)

    # Split the labels
    train_labels, val_labels = train_test_split(
                                    labels,
                                    test_size=val_size,
                                    shuffle=False)

    # Create the DataLoaders
    train_ids = train_ids.to(device)
    train_masks = train_masks.to(device)
    train_labels = train_labels.to(device)
    val_ids = val_ids.to(device)
    val_masks = val_masks.to(device)
    val_labels = val_labels.to(device)

    train_data = TensorDataset(train_ids, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    val_data = TensorDataset(val_ids, val_masks, val_labels)
    if uncertainty_flag:
        test_dataloader = DataLoader(val_data, batch_size=1)
    else:
        test_dataloader = DataLoader(val_data, batch_size=16)

    return train_dataloader, test_dataloader