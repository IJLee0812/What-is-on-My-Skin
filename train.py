import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import initialize_model

# Data Loading
base_dir = "./results"
dataset = torch.load(os.path.join(base_dir, "dataset.pth"))
data_transforms = torch.load(os.path.join(base_dir, "data_transforms.pth"))

# Device Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# K-Fold Setting
k_folds = 5
kf = KFold(n_splits = k_folds, shuffle = True, random_state = 42)


# Function for model train & evaluation
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, EPOCH):

    # - - - init - - -
    train_loss_history = []
    val_loss_history = []

    train_acc_history = []
    val_acc_history = []

    print("Start Training...")

    best_loss = float('inf')
    best_model_wts = model.state_dict() # Save the best model weights here

    patience_counter = 0
    patience = 5 # Early Stopping patience(!= Scheduler patience)
    # - - - - - - - - -


    for epoch in range(EPOCH):
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1} / {EPOCH}, Learning Rate : {current_lr:.6f}')
        print('- ' * 15)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train() # for train
                dataloader = train_loader
            else:
                model.eval() # for 'validation'(calculate validation loss)
                dataloader = val_loader

            running_loss = 0.0 # float
            running_corrects = 0 # integer

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'): # using autograd if train-mode
                    outputs = model(inputs) # inputs : from dataloader
                    _, preds = torch.max(outputs, 1) # preds : find 'index'
                    loss = criterion(outputs, labels) # outputs : prediction / labels : answer

                    if phase == 'train': # shouldn't occur gradient update when validation mode
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0) # for epoch_loss
                running_corrects += torch.sum(preds == labels.data) # for epoch_acc

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = float(running_corrects) / len(dataloader.dataset)

            print(f'{phase} Loss : {epoch_loss : .4f} Acc: {epoch_acc : .4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else: # phase == 'validation'
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                scheduler.step(epoch_loss) # for LR Scheduling

                # Early Stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict() # Save best model weights
                    patience_counter = 0 # If loss decreases, init counter
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early Stopping at epoch {epoch + 1}")
                        print('- ' * 15)
                        model.load_state_dict(best_model_wts) # Load best model weights before returning
                        # Plotting <train / validation> loss
                        plt.figure()
                        plt.plot(train_loss_history, label = 'Train Loss')
                        plt.plot(val_loss_history, label = 'Validation Loss')
                        plt.xlabel('Epoch') ; plt.ylabel('Loss')
                        plt.legend(loc = 'best')
                        plt.title('Training and Validation Loss')

                        # Plotting <train / validation> accuracy
                        plt.figure()
                        plt.plot(train_acc_history, label = 'Train Accuracy')
                        plt.plot(val_acc_history, label = 'Validation Accuracy')
                        plt.xlabel('Epoch') ; plt.ylabel('Accuracy')
                        plt.legend(loc = 'best')
                        plt.title('Training and Validation Accuracy')
                        return model

        print('- ' * 15) # No Early Stopping

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plotting <train / validation> loss
    plt.figure()
    plt.plot(train_loss_history, label = 'Train Loss')
    plt.plot(val_loss_history, label = 'Validation Loss')
    plt.xlabel('Epoch') ; plt.ylabel('Loss')
    plt.legend(loc = 'best')
    plt.title('Training and Validation Loss')

    # Plotting <train / validation> accuracy
    plt.figure()
    plt.plot(train_acc_history, label = 'Train Accuracy')
    plt.plot(val_acc_history, label = 'Validation Accuracy')
    plt.xlabel('Epoch') ; plt.ylabel('Accuracy')
    plt.legend(loc = 'best')
    plt.title('Training and Validation Accuracy')

    return model


def main():
    results = []
    EPOCH = 100

    # 5-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f'Fold {fold + 1}')

        # dataset splitting using Subset
        train_subsampler = Subset(dataset, train_idx)
        val_subsampler = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subsampler, batch_size = 32, shuffle = True, num_workers = 2)
        val_loader = DataLoader(val_subsampler, batch_size = 32, shuffle = True, num_workers = 2)

        # model initialization
        model = initialize_model(len(dataset.classes))
        model = model.to(device)

        # define loss function, optimizer and scheduler
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01) # Using weight decay
        scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 3)

        # - - - model train & validation part - - -
        trained_model = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, EPOCH)

        # Save the best model for each fold
        torch.save(trained_model.state_dict(), os.path.join(base_dir, f"best_model_fold_{fold + 1}.pth"))

        # final evaluation on validation dataset
        trained_model.eval() # set model to evaluation mode
        running_corrects = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = trained_model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = float(running_corrects) / len(val_loader.dataset)

        # save & output each fold results
        results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'labels': all_labels,
            'preds': all_preds
        })

    for result in results:
        print(f"Fold {result['fold']} - Accuracy: {result['accuracy']:.4f}")

    # computing confusion matrix, precision, recall and F1-Score
    all_labels = np.concatenate([result['labels'] for result in results])
    all_preds = np.concatenate([result['preds'] for result in results])

    # Visualization
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis] # Normalize(0 ~ 1)
    plt.figure(figsize = (9, 9))
    sns.heatmap(cm_normalized, annot = True, fmt = '.2f', cmap = 'Reds', xticklabels = dataset.classes, yticklabels = dataset.classes)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Model Predicted')
    plt.ylabel('True')
    plt.show()

    # 2. Precision, Recall, F1-Score of each class
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=np.arange(len(dataset.classes)))
    metrics_df = pd.DataFrame({
        'Class': dataset.classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    plt.figure(figsize = (7, 5))
    metrics_df.plot(kind='bar', x='Class', rot=45, figsize=(10, 6))
    plt.title('Classwise Precision, Recall, F1-Score')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
