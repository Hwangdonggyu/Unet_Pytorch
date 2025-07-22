import time
from data.database import DataProvider
from model.network_unet import Unet
from loss.unet_loss import BCEDiceLoss
import torch
import torch.optim as optim
from math import *
import os
import matplotlib.pyplot as plt
import numpy as np

# -- implementing... ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

result_dir = os.path.join('./result')
if not os.path.exists(result_dir):
    os.mkdir(os.path.join(result_dir, 'png'))
    os.mkdir(os.path.join(result_dir, 'npy'))

data_root = 'data_root'
pretrained_root = 'pretrained_root'
batch_size = 1

data_test = DataProvider(data_root)
dataset_size = data_test.n_data
test_iters = int(dataset_size)
print(f'#testing images = {dataset_size}')


model = Unet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load(pretrained_root))

criterion = BCEDiceLoss()

test_loss = 0

with torch.no_grad():

    for _ in range(test_iters):
        model.eval()
        batch_x, batch_y, path = data_test(batch_size)

        # GPU로 옮기기
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_x)
        mask = batch_y
        loss = criterion(output, mask)

        test_loss += loss.item()

        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()

        pred_np = pred.squeeze().cpu().numpy()
        gt_np = batch_y.squeeze().cpu().numpy()

        fname = os.path.basename(path[0]).replace('.nii', f'_slice{i}.png')

        plt.imsave(os.path.join(result_dir, 'png', fname), pred_np, cmap='gray')

        np.save(os.path.join(result_dir, 'npy', fname.replace('.png', '.npy')), pred_np)

    print(f"Total test loss: {test_loss:.4f}")