import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

class TrainingManager:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(device)

    def train_epoch(self, dataloader):
        total_loss, accuracy = 0, 0
        progress_bar = tqdm(range(len(dataloader)))
        
        self.model.train()
        for batch in dataloader:
            input_ids, token_type_ids, attention_mask, labels = batch.values()
            labels = labels.to(self.device)
            inputs = {'input_ids': input_ids.to(self.device),
                      'token_type_ids': token_type_ids.to(self.device),
                      'attention_mask': attention_mask.to(self.device)}
            
            # Forward pass
            logits = self.model(inputs)
            loss = self.loss_fn(logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # update progress bar
            progress_bar.update(1)
            
            total_loss += loss.item()
            accuracy += (logits.argmax(dim=1) == labels).sum().item()
        return total_loss / len(dataloader), accuracy / len(dataloader.dataset)
    
    def evaluate(self, dataloader):
        total_loss, accuracy = 0, 0
        all_preds = []
        
        self.model.eval()
        for batch in dataloader:
            input_ids, token_type_ids, attention_mask, labels = batch.values()
            labels = labels.to(self.device)
            inputs = {'input_ids': input_ids.to(self.device),
                      'token_type_ids': token_type_ids.to(self.device),
                      'attention_mask': attention_mask.to(self.device)}
            
            with torch.no_grad():
                logits = self.model(inputs)
                loss = self.loss_fn(logits, labels)
                
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            accuracy += (predictions == labels).sum().item()
            all_preds = np.concatenate((all_preds, predictions.cpu().numpy()))
            
        f1scores= f1_score(dataloader.dataset.labels, all_preds, average=None)
        mf1score = f1_score(dataloader.dataset.labels, all_preds, average='macro')
        return total_loss / len(dataloader), accuracy / (len(dataloader.dataset)), f1scores, mf1score