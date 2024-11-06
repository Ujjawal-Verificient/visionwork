import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
import copy

import matplotlib.pyplot as plt

def plot_training_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='green')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_acc(train_acc, val_acc):
    epochs = range(1, len(train_acc) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Training Acc', color='blue')
    plt.plot(epochs, val_acc, label='Validation Acc', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title('Training and Validation Acc')
    plt.legend()
    plt.tight_layout()
    plt.show()

class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.data.iloc[idx, 0])
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        # print(img_name)
        labels = self.data.iloc[idx, 1:].values.astype('float32')
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(labels)
        
        return image, labels

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# csv_file = 'data/finetune_annotations.csv'      
# img_dir = '/home/ajeet/codework/datasets/finetune_images_All_cts_cellphone_dataset/images'
csv_file = 'annotations.csv'
img_dir = '/home/ajeet/codework/datasets/finetune_images_All_cts_cellphone_dataset/test/images'
dataset = MultiLabelDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
val_dataset = MultiLabelDataset(csv_file="data/val10_finetune_annotations.csv", img_dir="/home/ajeet/codework/datasets/finetune_images_All_cts_cellphone_dataset/val/images", transform=transform)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
for param in model.parameters():
    param.requires_grad = False

# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 256), 
#     nn.ReLU(),                 
#     nn.Dropout(0.5),          
#     nn.Linear(256, 4)         
# )

for param in model.fc.parameters():
    param.requires_grad = True

# for param in model.layer4.parameters():
#     param.requires_grad = True

print("Total Parameters:", sum(p.numel() for p in model.parameters()))
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable_params: {trainable_params}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device = "cpu"
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion, optimizer, num_epochs=10):
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = [] 
    train_accuracies = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        model.train()
        # running_loss = 0.0
        training_loss = 0.0
        all_preds = []
        all_labels = []
        correct_preds = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # print(outputs)
            # print(labels)
            # print("----")
            print()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            outputs = torch.sigmoid(outputs)
            predicted = torch.round(outputs)
            correct_preds += torch.sum(torch.all(torch.eq(predicted, labels), dim=1)).item()
            total_samples += labels.size(0)

        train_loss = training_loss / len(dataloader)
        train_losses.append(train_loss)
        train_acc = correct_preds / total_samples * 100 
        train_accuracies.append(train_acc)

        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_samples = 0
        with torch.no_grad():
            for input, labels in val_dataloader:
                inputs = input.to(device)
                labels = labels.float().to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                outputs = torch.sigmoid(outputs)
                predicted = torch.round(outputs)
                # print(outputs)
                # print(labels)
                total_samples += labels.size(0)
                correct_preds += torch.sum(torch.all(torch.eq(predicted, labels), dim=1)).item()


        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        val_acc = correct_preds / total_samples * 100
        val_accs.append(val_acc)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f}  Validation Loss: {val_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%")

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), f'finetuned_models/best_model_epoch_{epoch + 1}.pth')


    plot_training_loss(train_losses, val_losses)
    plot_training_acc(train_accuracies, val_accs)



            # Track statistics
        #     print(inputs.size(0))
        #     running_loss += loss.item()
        #     preds = torch.sigmoid(outputs) > 0.5  # Sigmoid threshold for multi-label
        #     # print(preds)
        #     all_preds.append(preds.cpu().numpy())
        #     all_labels.append(labels.cpu().numpy())

        # epoch_loss = running_loss / len(dataset)
        # train_losses.append(epoch_loss)
        # all_preds = np.vstack(all_preds)
        # all_labels = np.vstack(all_labels)

        # # print(all_labels)
        # # print(all_preds)
        # # Calculate accuracy (or other metric)
        # epoch_acc = accuracy_score(all_labels, all_preds)
        # train_accuracies.append(epoch_acc)
        # print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # if epoch_acc > best_acc:
        #     best_acc = epoch_acc
        #     # best_model_wts = copy.deepcopy(model.state_dict())

    #     print()

    # print(f"Best val Acc: {best_acc:.4f}")
    # plot_training_metrics(train_losses, train_accuracies)
    # # model.load_state_dict(best_model_wts)
    # return model

# 7. Train the Model
num_epochs = 20
model = train_model(model, criterion, optimizer, num_epochs=num_epochs)

# torch.save(model.state_dict(), "multi_label_resnet50.pth")