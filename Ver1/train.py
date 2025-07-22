import time
from data.database import DataProvider
from model.network_unet import Unet
from loss.unet_loss import BCEDiceLoss
import torch
import torch.optim as optim
from math import *
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_root = 'data_roott'
batch_size = 4
epoch = 5

data_train = DataProvider(data_root)
dataset_size = data_train.n_data
training_iters = int(ceil(dataset_size/float(batch_size)))
print('#training images = %d' % dataset_size)

total_steps = 0
model = Unet(n_channels=1, n_classes=1).to(device)
check_dir = './check_dir'
if not os.path.exists(check_dir):
  os.makedirs(check_dir)

criterion = BCEDiceLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for _ in range(1, epoch+1):
  epoch_start_time = time.time()

  epoch_loss = 0

  for _ in range(training_iters):
    model.train()
    batch_x, batch_y, path = data_train(batch_size)

    # GPU로 옮기기
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)


    optimizer.zero_grad()
    output = model(batch_x)
    mask = batch_y
    loss = criterion(output, mask)
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
  
  print(f"Epoch : {epoch}, Loss : {epoch_loss}, Time : {time.time()-epoch_start_time}")
  total_steps += 1

torch.save(model.state_dict(), os.path.join(check_dir, 'unet_epoch%d.pth' % epoch))
