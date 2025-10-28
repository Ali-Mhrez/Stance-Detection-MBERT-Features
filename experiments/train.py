import os
import torch
import argparse
from torch.optim import AdamW
from ..models.bert_cnn import BERTCNN
from ..models.bert_lstm import BERTLSTM
from ..models.bert_cnn_lstm import BERTCNNLSTM
from torch.utils.data import DataLoader
from ..data.custom_dataset import CustomDataset
from transformers import AutoTokenizer, AutoModel
from ..utility.utils import write_results, set_seed
from ..training.training_manager import TrainingManager
from ..data.dataset_reader import AraStanceReader, UnifiedFCReader, stance_to_int

if __name__ == "__main__":
    
    lr = 1e-4
    no_reps = 5
    no_epochs = 25
    batch_size = 64
    sequence_length = 256
    seeds = [42, 43, 44, 45, 46]
    
    parser = argparse.ArgumentParser(description="Run fine tuning experiments.")
    parser.add_argument('--dataset', type=str, choices=['arastance', 'unifiedfc'], required=True, help='Dataset to use for fine-tuning.')
    parser.add_argument('--model-name', type=str, choices=['bertcnn', 'bertlstm', 'bertcnnlstm'], required=True, help='Model name to use.')
    parser.add_argument('--setting', type=str, choices=['last-layer', 'last-4-layers'], required=True, help='BERT layer setting.')
    args = parser.parse_args()
    dataset = args.dataset
    setting = args.setting
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "raw_data")
    
    print("Loading data and labels...")
    if dataset == 'unifiedfc':
        raw_train = UnifiedFCReader(os.path.join(data_path, "unifiedfc-main/"), folds=['Fold_1', 'Fold_2', 'Fold_3'])
        raw_val = UnifiedFCReader(os.path.join(data_path, "unifiedfc-main/"), folds=['Fold_4'])
        raw_test = UnifiedFCReader(os.path.join(data_path, "unifiedfc-main/"), folds=['Fold_5'])
    elif dataset == 'arastance':
        raw_train = AraStanceReader(os.path.join(data_path, 'arastance-main/train.jsonl'))
        raw_val = AraStanceReader(os.path.join(data_path, 'arastance-main/dev.jsonl'))
        raw_test = AraStanceReader(os.path.join(data_path, 'arastance-main/test.jsonl'))
    print("Data and labels loaded successfully.")
     
    print("Loading pretrained model and tokenizer...")
    checkpoint = 'bert-base-multilingual-cased'
    pretrained_model = AutoModel.from_pretrained(checkpoint, output_hidden_states=True, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    def tokenize(instance):
        return tokenizer(instance[0], instance[1], truncation=True, padding='max_length', max_length=sequence_length)
    print("Pretrained model and tokenizer loaded successfully.")
    
    print("Tokenizing data...")
    train_labels = [stance_to_int[stance] for stance in raw_train.stances]
    val_labels = [stance_to_int[stance] for stance in raw_val.stances]
    test_labels = [stance_to_int[stance] for stance in raw_test.stances]
    
    train_claims = list(map(raw_train.claims.__getitem__, raw_train.article_claim))
    val_claims = list(map(raw_val.claims.__getitem__, raw_val.article_claim))
    test_claims = list(map(raw_test.claims.__getitem__, raw_test.article_claim))
    
    train_encodings = tokenize((train_claims, raw_train.articles))
    val_encodings = tokenize((val_claims, raw_val.articles))
    test_encodings = tokenize((test_claims, raw_test.articles))
    
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    print("Tokenization completed.")
    
    print("Starting model training...")
    for rep in range(no_reps):
        
        print(f"\n--- Repetition {rep+1} of {no_reps} ---")
        set_seed(seeds[rep])
        
        print("Creating DataLoaders...")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print("DataLoaders created successfully.")
        
        print("Initializing model, optimizer, and loss function...")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if args.model_name == 'bertcnn':
            model = BERTCNN(pretrained_model, sequence_length, setting)
        elif args.model_name == 'bertlstm':
            model = BERTLSTM(pretrained_model, setting)
        elif args.model_name == 'bertcnnlstm':
            model = BERTCNNLSTM(pretrained_model, sequence_length, setting)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=lr)
        manager = TrainingManager(model, optimizer, loss_fn, device)
        print("Model, optimizer, and loss function initialized successfully.")
        
        train_accuracies, train_losses = [], []
        eval_accuracies, eval_losses = [], []
        for epoch in range(no_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # --- Training Phase ---
            train_loss, train_accuracy = manager.train_epoch(train_dataloader)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)

            # --- Evaluation Phase ---
            eval_loss, eval_accuracy, eval_f1scores, eval_mf1score = manager.evaluate(dev_dataloader)
            eval_accuracies.append(eval_accuracy)
            eval_losses.append(eval_loss)

            print(f"Train_loss: {train_loss:.3f}, Train_acc: {train_accuracy:.3f},",
                f"Val_loss: {eval_loss:.3f}, Val_accuracy: {eval_accuracy:.3f}",
                "\n-------------------------------")
            
        test_loss, test_accuracy, test_f1scores, test_mf1score = manager.evaluate(test_dataloader)
        
        results_path = os.path.join(os.path.dirname(__file__), "..", "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            
        write_results(
            filename=os.path.join(results_path, f"results.csv"),
            run_id=rep+1,
            eval_loss=eval_losses[-1],
            eval_acc=eval_accuracies[-1],
            eval_f1=eval_f1scores,
            eval_mf1=eval_mf1score,
            test_loss=test_loss,
            test_acc=test_accuracy,
            test_f1=test_f1scores,
            test_mf1=test_mf1score
        )