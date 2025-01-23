from tqdm import tqdm
import time
import torch
import torch.nn as nn
import wandb

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    model.eval()
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    dt = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path = f"{save_path}checkpoint_{dt}_epoch{epoch + 1}.pt"
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")
    model.train()

def train_model(model, train_dataloader, optimizer, scheduler, EPOCHS, model_save_path):
    wandb.init(
    project="bert-finetuning-tutorial",
    config={
        "learning_rate": optimizer.param_groups[0]['lr'],
        "architecture": "BERT",
        "epochs": EPOCHS
    })
    wandb.watch(model, log='all', log_freq=10)

    model.train()

    for epoch in tqdm(range(0, EPOCHS), desc='Epochs'):
        training_loss = 0

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):

            batch_token_ids = batch[0]
            batch_attention_mask = batch[1]
            batch_labels = batch[2]
            
            model.zero_grad()

            loss, logits = model(
                batch_token_ids,
                token_type_ids = None,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                return_dict=False)

            loss = loss.mean()
            batch_loss = loss.item()
            training_loss += batch_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            wandb.log({"batch_loss": batch_loss,
                       "learning_rate": optimizer.param_groups[0]['lr']
            })

        average_train_loss = training_loss / len(train_dataloader)
        wandb.log({"epoch_loss": average_train_loss,
                   "epochs": epoch + 1
        })

        save_checkpoint(model, optimizer, scheduler, epoch, average_train_loss, model_save_path)
    
    # model.eval()
    # torch.save(model.state_dict(), model_save_path)
    print("Training complete")
    print(f"Model saved as {model_save_path}")
    print(f"Average training loss: {average_train_loss}")