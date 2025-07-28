import time
from data.dataloader import LiTSDataset
from model.network_unet import Unet
from loss.unet_loss import BCEDiceLoss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from math import *
import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_root = 'test_data_root'
pretrained_root = 'pretrained_root'
batch_size = 4

data_test = LiTSDataset(data_root, mode='train')
dataset_size = len(data_test)
print('#test images = %d' % dataset_size)

test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2)

model = Unet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load(pretrained_root))

criterion = BCEDiceLoss()

test_loss = 0

with torch.no_grad():
    model.eval()
    test_start_time = time.time()
    
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_x)
        loss = criterion(output, batch_y)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
  
print(f"Test Loss: {avg_test_loss:.4f}, Time: {time.time() - test_start_time:.2f}s")
