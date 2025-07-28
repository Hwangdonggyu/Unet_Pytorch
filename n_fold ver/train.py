import time
from data.dataloader import LiTSDataset
from model.network_unet import Unet
from loss.unet_loss import BCEDiceLoss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from math import *
import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_root = 'train data root'
batch_size = 4
epochs = 10

data_train = LiTSDataset(data_root, mode='train')
dataset_size = len(data_train)
print('#training images = %d' % dataset_size)

kfold = KFold(n_splits = 5, shuffle=True)

for fold, (train_ids, val_ids) in enumerate(kfold.split(data_train)):

  print(f"Fold {fold +1}/{kfold}")
  writer = SummaryWriter(log_dir=f"runs/fold{fold+1}")

  train_subsample = Subset(data_train, train_ids)
  val_subsample = Subset(data_train, val_ids)

  train_loader = DataLoader(train_subsample, batch_size=batch_size, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_subsample, batch_size=batch_size, shuffle=False, num_workers=2)

  model = Unet(n_channels=1, n_classes=1).to(device)
  check_dir = './check_dir'
  if not os.path.exists(check_dir):
    os.makedirs(check_dir)

  criterion = BCEDiceLoss()

  optimizer = optim.Adam(model.parameters(), lr=1e-4)

  for epoch in range(1, epochs+1):
    epoch_start_time = time.time()

    train_loss = 0
    loop = tqdm.tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch}/{epochs}", leave=False)

    for i, (batch_x, batch_y) in enumerate(loop):
      model.train()

      # GPU로 옮기기
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)

      output = model(batch_x)
      mask = batch_y
      loss = criterion(output, mask)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      loop.set_postfix(loss=loss.item())  # 현재 배치 loss 표시

      if i%50==0:
        print(f"Fold {fold+1}, Epoch {epoch}, Train Loss: {loss:.4f}")

      if i == 0:
        writer.add_image("Train/Output", torch.sigmoid(output[0]), epoch, dataformats='CHW')
        writer.add_image("Train/Mask", batch_y[0], epoch, dataformats='CHW')

    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar(f"Loss/train_fold{fold}", avg_train_loss, epoch)

    print(f"Fold {fold+1}, Epoch {epoch}, Avg Train Loss: {avg_train_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s")

    model.eval()
    val_loss = 0
    with torch.no_grad():
      for i, (val_x, val_y) in enumerate(val_loader):
        val_x = val_x.to(device)
        val_y = val_y.to(device)

        val_output = model(val_x)
        loss = criterion(val_output, val_y)
        val_loss += loss.item()

        if i == 0:
          writer.add_image("Val/Output", torch.sigmoid(val_output[0]), epoch, dataformats='CHW')
          writer.add_image("Val/Mask", val_y[0], epoch, dataformats='CHW')

      avg_val_loss = val_loss / len(val_loader)
      writer.add_scalar(f"Loss/val_fold{fold}", avg_val_loss, epoch)
      print(f"Fold {fold+1}, Epoch {epoch}, Val Loss: {avg_val_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s")

torch.save(model.state_dict(), os.path.join(check_dir, 'unet_epoch%d.pth' % epoch))