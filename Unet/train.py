import os
from time import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.data import DriveDataset
from src.model import build_unet
from src.loss import DiceLoss, DiceBCELoss
from src.utils import seeding, create_dir, epoch_time

""" Training Model """
def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    epoch_loss = epoch_loss / len(loader)
    return epoch_loss
    
def evaluate(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(loader)
    return epoch_loss
    
if __name__ == '__main__':
    """ Seeding """
    seeding(42)
    
    """ Directories """
    create_dir('checkpoint')
    
    """ Load Dataset """
    train_x = sorted(glob('new_dataset/train/image/*'))[:20]
    train_y = sorted(glob('new_dataset/train/mask/*'))[:20]
    
    test_x = sorted(glob('new_dataset/test/image/*'))
    test_y = sorted(glob('new_dataset/test/mask/*'))
    
    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = 'checkpoint/checkpoint.pth'
    
    """ Dataset Loader """
    train_dataset = DriveDataset(train_x, train_y)
    test_dataset = DriveDataset(test_x, test_y)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.device_count())
    model = build_unet()
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
        
    """ Training Model """
    best_test_loss = float('inf')
    
    """ Transfer Learning Part """
    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    for epoch in range(num_epochs):
        start_time = time()
        
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        test_loss = evaluate(model, test_loader, optimizer, loss_fn, device)
        
        """ Saving Model """
        if test_loss < best_test_loss:
            data_str = f'Valid loss improved from {best_test_loss:2.4f} to {test_loss:2.4f}. Saving checkpoint: {checkpoint_path}'
            print(data_str)
            
            best_test_loss = test_loss
            torch.save(model.state_dict(), checkpoint_path)
                    
        end_time = time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        data_str = f'Epoch: {epoch+1:02} | Epoch time: {epoch_mins}m {epoch_secs}s \n'
        data_str += f'Train loss: {train_loss:.3f}\n'
        data_str += f'Test loss: {test_loss:.3f}\n'
        
        print(data_str)