import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import librosa
from transformers import Wav2Vec2FeatureExtractor
import numpy as np

CFG = {
    'SR': 16_000,
    'BATCH_SIZE': 8,
    'TOTAL_BATCH_SIZE': 42,
    'N_MFCC': 32, # Melspectrogram 벡터를 추출할 개수
    'SEED': 42,
    'EPOCHS': 1,
    'LR': 1e-4
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def speech_file_to_array_fn(df):
    feature = []
    for path in tqdm(df['path']):
        speech_array, _ = librosa.load(path, sr=CFG['SR'])
        print(f"sfaf : speech_array, _ = librosa.load(path, sr=CFG['SR']), speech_array = {speech_array}")
        feature.append(speech_array)
        print("sfaf : feature.append(speech_array)")
    return feature

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y, processor):
        self.x = x
        self.y = y
        self.processor = processor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        input_values = self.processor(self.x[idx], sampling_rate=CFG['SR'], return_tensors='pt', padding=True).input_values
        if self.y is not None:
            print("CustomDataSet : __getitem__ : if self.y is not None:")
            return input_values.squeeze(), self.y[idx]
        else:
            print("CustomDataSet : __getitem__ : else:")
            return input_values.squeeze()



def collate_fn(batch):
    x, y = zip(*batch)
    print(f"collate_fn : x, y = zip(*batch), x = {x}, y = {y}")
    x = pad_sequence([torch.tensor(xi) for xi in x], batch_first=True)
    print("collate_fn : x = pad_sequence([torch.tensor(xi) for xi in x], batch_first=True)")
    print(f"x = {x}")
    y = pad_sequence([torch.tensor([yi]) for yi in y], batch_first=True)  # Convert scalar targets to 1D tensors
    print("collate_fn : y = pad_sequence([torch.tensor([yi]) for yi in y], batch_first=True)")
    print(f"y = {y}")
    return x, y


def create_data_loader(dataset, batch_size, shuffle, collate_fn, num_workers=0):
    print("create_data_loader : Data Loaded")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)


def validation(model, valid_loader, criterion):
    model.eval()
    val_loss = []

    total, correct = 0, 0
    test_loss = 0

    with torch.no_grad():
        for x, y in tqdm(iter(valid_loader)):
            if y != None:
                print(f"validation : x = {x}")
                x = x.to(device)
                print(f"validation : x.to(device) = {x}")
                print(f"validation : y = {y}")
                y = y.flatten().to(device)
                print(f"validation : y.flatten().to(device) = {y}")
                output = model(x)
                print(f"validation : output = model(x) = {output}")
                loss = criterion(output, y)
                print(f"validation : loss = criterion(output, y) = {loss}")

                val_loss.append(loss.item())
                print("validation : val_loss.append(loss.item())")

                test_loss += loss.item()
                print("validation : test_loss += loss.item()")
                _, predicted = torch.max(output, 1)
                print("validation : _, predicted = torch.max(output, 1)")
                total += y.size(0)
                print("validation : total += y.size(0)")
                correct += predicted.eq(y).cpu().sum()
                print("validation : correct += predicted.eq(y).cpu().sum()")
            else:
                pass

    accuracy = correct / total
    print("validation : accuracy = correct / total")
    avg_loss = np.mean(val_loss)
    print("validation : avg_loss = np.mean(val_loss)")

    return avg_loss, accuracy


def train(model, train_loader, valid_loader, optimizer, scheduler):
    accumulation_step = int(CFG['TOTAL_BATCH_SIZE'] / CFG['BATCH_SIZE'])
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_model = None
    best_acc = 0
    epoch_count = 1
    for epoch in range(1, CFG['EPOCHS']+1):
        print(f'train : epoch count = {epoch_count} th count')
        train_loss = []
        model.train()
        train_loader_count = 1
        for i, (x, y) in enumerate(tqdm(train_loader)):
            print(f"train : epoch_count = {epoch_count}, train_loader_count = {train_loader_count}")
            print(f"validation : x = {x}")
            x = x.to(device)
            print(f"train : x.to(device) = {x}")
            print(f"train : y = {y}")
            y = y.flatten().to(device)
            print(f"train : y.flatten().to(device) = {y}")
            print("train : x, y moved to device")
            optimizer.zero_grad()
            print("train : optimizer.zero_grad()")
            output = model(x)
            print(f"train : output = model(x) = {output}")
            loss = criterion(output, y)
            print(f"train : loss = criterion(output, y) = {loss}")
            loss.backward()
            print("train : loss.backward()")

            if (i + 1) % accumulation_step == 0:
                print("train : if (i + 1) % accumulation_step == 0:")
                optimizer.step()
                print("train : optimizer.step()")
                optimizer.zero_grad()
                print("optimizer.zero_grad()")
            train_loader_count += 1
            train_loss.append(loss.item())
            print("train : train_loss.append(loss.item())")

        avg_loss = np.mean(train_loss)
        print(f"train : avg_loss = np.mean(train_loss) = {avg_loss}")
        valid_loss, valid_acc = validation(model, valid_loader, criterion)
        print("train : valid_loss, valid_acc = validation(model, valid_loader, criterion)")

        if scheduler is not None:
            scheduler.step(valid_acc)
            print("train : if scheduler is not None: scheduler.step(valid_acc)")

        if valid_acc > best_acc:
            print("train : if valid_acc > best_acc:")
            best_acc = valid_acc
            print(f"train : best_acc = valid_acc = {best_acc}")
            best_model = model
            print(f"train : best_model = model = {best_model}")
        epoch_count += 1
        print(f'train : epoch:[{epoch}] train loss:[{avg_loss:.5f}] valid_loss:[{valid_loss:.5f}] valid_acc:[{valid_acc:.5f}]')

    print(f'train : best_acc:{best_acc:.5f}')

    return best_model

'''

import torch
import torch.utils.data as data

class CustomDataSet(data.Dataset):
    def __init__(self, x, y, processor):
        self.x = x
        self.y = y
        self.processor = processor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        input_values = self.processor(self.x[idx], sampling_rate=CFG['SR'], return_tensors="pt", padding=True).input_values
        if self.y is not None:
            return input_values.squeeze(), self.y[idx]
        else:
            return input_values.squeeze()
