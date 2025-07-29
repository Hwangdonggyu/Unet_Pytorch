import time
from data.dataloader import LiTSDataset
from model.network_unet import Unet
from loss.unet_loss import BCEDiceLoss
from metric.metric import Metric
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from math import *
import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_root = 'test_data_root'
pretrained_root = 'pretrained_root'
batch_size = 4

data_test = LiTSDataset(data_root, mode='test')
dataset_size = len(data_test)
print('#test images = %d' % dataset_size)

writer = SummaryWriter(log_dir='runs/test_logs')

test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2)

model = Unet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load(pretrained_root))

criterion = BCEDiceLoss()
metric = Metric()

test_loss = 0

with torch.no_grad():
    model.eval()
    test_start_time = time.time()
    
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_x)
        loss = criterion(output, batch_y)
        test_loss += loss.item()
        test_dice_score += dice_score.item()
        test_pixel_acc += pixel_acc

        if (i % 10) == 0:
            writer.add_image("Test/Input", batch_x[0], i, dataformats='CHW')
            writer.add_image("Test/Mask", batch_y[0], i, dataformats='CHW')
            writer.add_image("Test/Output", torch.sigmoid(output[0]), i, dataformats='CHW')

avg_test_loss = test_loss / len(test_loader)
avg_test_dice_score = test_dice_score / len(test_loader)
avg_test_pixel_acc = test_pixel_acc / len(test_loader)

print(f"Avg Test Loss: {avg_test_loss:.4f}, Avg Test Dice Score: {avg_test_dice_score:.4f}, Avg Test Pixel Accuracy: {avg_test_pixel_acc:.4f} Time: {time.time() - test_start_time:.2f}s")

writer.add_scalar("Loss/test", avg_test_loss, 0)

writer.close()