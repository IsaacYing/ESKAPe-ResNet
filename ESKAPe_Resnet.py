import os
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import random
import string
import pickle as pk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.models import  ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
from torchvision import models as torch_models
from torch.optim import Adam
from torch.utils.data import Subset
from torch.nn import DataParallel
from PIL import Image

def generate_random_string():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(3))

random_string = generate_random_string()

# Random_seed
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Time and Log
def record(content):
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M")
    os.makedirs('logs', exist_ok=True)
    with open(f"logs/{random_string}_log.txt", "a") as f:
        f.write(time_str)
        f.write("\n")
        f.write(content)
        f.write("\n")

# Vividict
class Vividict(dict):
    def __missing__(self,key):
        value = self[key] = type(self)()
        return value

def initialize_model(num_classes=6):
    model = torch_models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.file_paths = []
        self.cell_records = []
        self.root_dir = root_dir
        self.transform = transform
        self.current_file = None
        self.current_batch = None
        
        self.classes1 = {'ECO_MG1655': 0, 'S_ATCC43300': 1, 'Kp_ATCC43816': 2, 
                         'A_ATCC19606': 3, 'P_PA01': 4, 'P_PA14': 4, 
                         'P_PAK': 4, 'VAE_21B188': 5}
        self.classes2 = {'A_ABA': 3, 'A_AB09': 3}
        
        self._collect_cell_records()
        
    def _collect_cell_records(self):
        for cls_name, label in self.classes1.items():
            class_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue
            for batch_file in os.listdir(class_dir):
                if batch_file.lower().endswith('.pkl'):
                    batch_path = os.path.join(class_dir, batch_file)
                    try:
                        with open(batch_path, 'rb') as f:
                            batch_cell = pk.load(f)
                        for cid in batch_cell:
                            self.file_paths.append(batch_path)
                            self.cell_records.append((batch_path, cid, label))
                    except Exception as e:
                        record(f"Failed to load file header: {batch_path}, error:{str(e)}")
                        
        for cls_name, label in self.classes2.items():
            class_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue
            for batch_file in os.listdir(class_dir):
                if batch_file.lower().endswith('.pkl'):
                    batch_path = os.path.join(class_dir, batch_file)
                    try:
                        with open(batch_path, 'rb') as f:
                            batch_cell = pk.load(f)
                        for cid in batch_cell:
                            self.file_paths.append(batch_path)
                            self.cell_records.append((batch_path, cid, label))
                    except Exception as e:
                        record(f"Failed to load file header: {batch_path}, error:{str(e)}")
        self.unique_file_paths = list(set(self.file_paths))
        record(f"Dataset include {len(self.cell_records)} cells, {len(self.unique_file_paths)} files.")

    def __len__(self):
        return len(self.cell_records)

    def _load_file_if_needed(self, file_path):
        if self.current_file != file_path:
            self.current_batch = None
            try:
                with open(file_path, 'rb') as f:
                    self.current_batch = pk.load(f)
                self.current_file = file_path
            except Exception as e:
                record(f"Can not load files: {file_path}, Error: {str(e)}")
                return False
        return True

    def __getitem__(self, idx):
        file_path, cid, label = self.cell_records[idx]
        
        if not self._load_file_if_needed(file_path) or cid not in self.current_batch:
            record(f"Can not load files or cell is not exist: {file_path}, cid: {cid}")
            placeholder = np.ones((224, 224), dtype=np.float32) * 0.5
            cell_img = placeholder
        else:
            cell_img = self.current_batch[cid]
        
        cell_img = (cell_img - cell_img.min()) / (cell_img.max() - cell_img.min())
        cell_img = (cell_img * 255).astype(np.uint8)

        img_pil = Image.fromarray(cell_img, mode='L')
        img_pil = img_pil.convert('RGB')

        return img_pil, label

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, patience=10):
    best_val_loss = float('inf')
    best_acc = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    counter = 0
    device = next(model.parameters()).device
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'val_f1': [], 'val_auc': [],
        'lr': []
    }
    
    with open(f'logs/{random_string}_learning_rate.txt', 'w') as f:
        f.write('Epoch,Learning Rate\n')
    
    with open(f'logs/{random_string}_training_metrics.txt', 'w') as f:
        f.write('Epoch,Train Loss,Train Acc,Val Loss,Val Acc,Val F1,Val AUC\n')
    
    with open(f'logs/{random_string}_training_log.txt', 'w') as f:
        f.write('Epoch,Start Time,End Time,Train Time,Val Time\n')
    
    record("Training start...")
    for epoch in range(num_epochs):
        epoch_start = datetime.datetime.now()
        record(f"Epoch {epoch+1}/{num_epochs}")
        record('-' * 10)
        
        phase = 'train'
        model.train()
        train_start = datetime.datetime.now()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        train_end = datetime.datetime.now()
        train_duration = (train_end - train_start).total_seconds()
        
        phase = 'val'
        model.eval()
        val_start = datetime.datetime.now()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
        
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
            else:
                acc = 0
            record(f"Class {i} Accuracy: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            epoch_auc = 0.0
            record(f"AUC calculate error, set to 0.0")
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())
        history['val_f1'].append(epoch_f1)
        history['val_auc'].append(epoch_auc)
        
        if scheduler:
            scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        val_end = datetime.datetime.now()
        val_duration = (val_end - val_start).total_seconds()
        epoch_end = datetime.datetime.now()
        
        record(f'Train Loss: {history["train_loss"][-1]:.4f} Acc: {history["train_acc"][-1]:.4f}')
        record(f'Val Loss: {history["val_loss"][-1]:.4f} Acc: {history["val_acc"][-1]:.4f}')
        record(f'F1: {epoch_f1:.4f} AUC: {epoch_auc:.4f}')
        record(f'Current LR: {current_lr:.6f}')
        
        with open(f'logs/{random_string}_training_metrics.txt', 'a') as f:
            f.write(f'{epoch+1},{history["train_loss"][-1]:.6f},{history["train_acc"][-1]:.6f},'
                    f'{history["val_loss"][-1]:.6f},{history["val_acc"][-1]:.6f},'
                    f'{epoch_f1:.6f},{epoch_auc:.6f}\n')
        
        with open(f'logs/{random_string}_learning_rate.txt', 'a') as f:
            f.write(f'{epoch+1},{current_lr:.6f}\n')
        
        with open(f'logs/{random_string}_training_log.txt', 'a') as f:
            f.write(f'{epoch+1},{epoch_start.strftime("%Y-%m-%d %H:%M:%S")},'
                    f'{epoch_end.strftime("%Y-%m-%d %H:%M:%S")},'
                    f'{train_duration},{val_duration}\n')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.savefig(f'logs/{random_string}_confusion_matrix_epoch_{epoch+1}.png')
            plt.close()
            record(f"Save confusion_matrix: logs/{random_string}_confusion_matrix_epoch_{epoch+1}.png")
        
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_val_acc = epoch_acc.item()
            best_f1 = epoch_f1
            best_auc = epoch_auc
            counter = 0
            model_filename = f'logs/best_model_{random_string}_epoch{epoch+1}_loss{best_val_loss:.4f}.pth'
            torch.save(model.state_dict(), model_filename)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'best_f1': best_f1,
                'best_auc': best_auc,
                'history': history
            }
            torch.save(checkpoint, f'logs/checkpoint_{random_string}_epoch{epoch+1}.pth')
            record(f"Save best model, Validation loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.4f}")
        else:
            counter += 1
            record(f"Early Stop: {counter}/{patience}")
        
        if counter >= patience:
            record(f"Early stop at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
            record(f"Best Indicator - Accuracy: {best_val_acc:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
            break
    
    record("Training Finished!")
    return model, history

class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

record("Modules Loaded.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
record(f"Device: {device}")

os.makedirs('logs', exist_ok=True)



data_dir1 = "./20250707_model/1_main"
dataset1 = CustomDataset(root_dir=data_dir1, transform=None)
data_len1 = len(dataset1)
record(f"dataset1 length: {data_len1}")

data_dir2 = "./20250707_model/2_aba"
dataset2 = CustomDataset(root_dir=data_dir2, transform=None)
data_len2 = len(dataset2)
record(f"dataset2 length: {data_len2}")

data_dir3 = "./20250707_model/3_flu"
dataset3 = CustomDataset(root_dir=data_dir3, transform=None)
data_len3 = len(dataset3)
record(f"dataset3 length: {data_len3}")

data_dir4 = "./20250707_model/4"
dataset4 = CustomDataset(root_dir=data_dir4, transform=None)
data_len4 = len(dataset4)
record(f"dataset4 length: {data_len4}")

data_dir5 = "./20250707_model/251001"
dataset5 = CustomDataset(root_dir=data_dir5, transform=None)
data_len5 = len(dataset5)
record(f"dataset5 length: {data_len5}")

data_dir6 = "./20250707_model/251020"
dataset6 = CustomDataset(root_dir=data_dir6, transform=None)
data_len6 = len(dataset6)
record(f"dataset6 length: {data_len6}")

data_dir7 = "./20250707_model/251023"
dataset7 = CustomDataset(root_dir=data_dir7, transform=None)
data_len7 = len(dataset7)
record(f"dataset7 length: {data_len7}")

combined_dataset = ConcatDataset([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7])
data_len = len(combined_dataset)
record(f"all dataset length: {data_len}")


train_size = int(0.8 * len(combined_dataset))
val_size = int(0.1 * len(combined_dataset))
test_size = len(combined_dataset) - train_size - val_size

record("Split Started.")
train_dataset, val_dataset, test_dataset = random_split(
    combined_dataset, [train_size, val_size, test_size]
)
record("Split Ended.")

def count_labels_fast(subset, name, concat_dataset):
    indices = subset.indices
    labels  = []
    offsets = np.cumsum([0] + [len(d) for d in concat_dataset.datasets])
    for idx in indices:
        ds_idx = np.searchsorted(offsets[1:], idx, side='right')
        local_idx = idx - offsets[ds_idx]
        ds = concat_dataset.datasets[ds_idx]
        labels.append(ds.cell_records[local_idx][2])
    counter = Counter(labels)
    record(f"{name} Category distribution: {dict(sorted(counter.items()))}")
    return counter


train_counter = count_labels_fast(train_dataset, "Train", combined_dataset)
val_counter   = count_labels_fast(val_dataset,   "Val",   combined_dataset)
test_counter  = count_labels_fast(test_dataset,  "Test",  combined_dataset)

total_counter = train_counter + val_counter + test_counter
record(f"Overall dataset category distribution: {dict(sorted(total_counter.items()))}")


train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = TransformedDataset(train_dataset, transform=train_transform)
val_dataset = TransformedDataset(val_dataset, transform=val_test_transform)
test_dataset = TransformedDataset(test_dataset, transform=val_test_transform)


batch_size = 64
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
}

num_classes = 6
model = initialize_model(num_classes)
model = model.to(device)
model = DataParallel(model)
record("Cuda Succeed.")


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    patience=5, 
    verbose=True
)

model, history = train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler=scheduler,
    num_epochs=50,
    patience=10
)