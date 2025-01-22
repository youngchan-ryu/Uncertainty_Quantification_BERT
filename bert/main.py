import os
import torch

from preprocessing import preprocess_dataset
from tokenizer import tokenize_data, save_tokenized_data, load_tokenized_data
from dataloader import split_load_data
from model import init_model, init_optimizer, init_loss_function, init_scheduler
from training import train_model
from evaluate import evaluate_accuracy

def main(output_file, force_update, model_save_path, train_flag, eval_flag, EPOCHS):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model save path not found")
    
    if not os.path.exists(output_file) or force_update:
        print("Preprocessing started")
        df = preprocess_dataset()

        token_ids, attention_masks, labels = tokenize_data(df)
        save_tokenized_data(token_ids, attention_masks, labels, output_file)
    else:
        print("Loaded Preprocessing data")
        token_ids, attention_masks, labels = load_tokenized_data(output_file)

    print("Tokenized Data ready")

    train_dataloader, val_dataloader = split_load_data(token_ids, attention_masks, labels, device)
    print("DataLoaders ready")

    model = init_model(device)
    optimizer = init_optimizer(model)
    loss_function = init_loss_function()
    scheduler = init_scheduler(optimizer, train_dataloader, EPOCHS)

    if train_flag:
        train_model(model, train_dataloader, optimizer, scheduler, EPOCHS, model_save_path)

    if eval_flag:
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"Model file not found at {model_save_path}")
        acc, loss = evaluate_accuracy(val_dataloader, model_save_path, device)
        print(f"Validation accuracy: {acc}")
        print(f"Validation loss: {loss}")

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize IMDB dataset")
    parser.add_argument('--tokenize-data-output', type=str, default='tokenized_data.pkl',
                        help="Path to save/load the tokenized dataset")
    parser.add_argument('--force-update', action='store_true',
                        help="Force re-tokenization and update the saved file")
    parser.add_argument('--model-save-path', type=str, default='checkpoints/', help="Path to save the trained model")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the model")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs to train the model")
    args = parser.parse_args()

    if not (args.train or args.evaluate):
        parser.error("You must specify either --train or --evaluate.")

    if args.train:
        if not args.epochs:
            parser.error("You must specify --epochs when training the model.")
        print("Training the model...")

    elif args.evaluate:
        print("Evaluating the model...")

    main(args.tokenize_data_output, args.force_update, args.model_save_path, args.train, args.evaluate, args.epochs)