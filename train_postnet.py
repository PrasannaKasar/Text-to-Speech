from preprocess import get_post_dataset, DataLoader, collate_fn_postnet
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

device = 'cuda'

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(model_postnet, test_loader, writer, epoch):
    """Test the model on the test dataset."""
    model_postnet.eval()
    test_loss = 0.0
    pbar = tqdm(test_loader, desc=f'Testing at epoch {epoch}')
    with torch.no_grad():
        for i, data in enumerate(pbar):
            mel, mag = data
            mel, mag = mel.to(device), mag.to(device)

            mag_pred = model_postnet(mel)

            loss = nn.L1Loss()(mag_pred, mag)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    writer.add_scalars('test_loss_per_epoch', {'loss': avg_test_loss}, epoch)
    print(f"Test Loss at epoch {epoch}: {avg_test_loss}")

def main():
    # Load dataset and split into train and test datasets
    dataset = get_post_dataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    model_postnet = nn.DataParallel(ModelPostNet().to(device))
    optimizer = torch.optim.Adam(model_postnet.parameters(), lr=hp.lr)
    writer = SummaryWriter()  # TensorBoard logs
    global_step = 0

    for epoch in range(hp.epochs):
        # Load train and test data
        train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, 
                                  collate_fn=collate_fn_postnet, drop_last=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False, 
                                 collate_fn=collate_fn_postnet, drop_last=False, num_workers=8)

        model_postnet.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Training at epoch {epoch}')

        for i, data in enumerate(pbar):
            global_step += 1

            # Adjust learning rate for warm-up phase
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)

            mel, mag = data
            mel, mag = mel.to(device), mag.to(device)

            # Forward pass
            mag_pred = model_postnet(mel)
            loss = nn.L1Loss()(mag_pred, mag)

            # Log training loss
            writer.add_scalars('training_loss', {'loss': loss.item()}, global_step)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_postnet.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Save checkpoints periodically
            if global_step % hp.save_step == 0:
                torch.save({'model': model_postnet.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(hp.checkpoint_path, f'checkpoint_postnet_{global_step}.pth.tar'))

        # Log average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalars('training_loss_per_epoch', {'loss': avg_train_loss}, epoch)
        print(f"Training Loss at epoch {epoch}: {avg_train_loss}")

        # Run the test loop at the end of each epoch
        test(model_postnet, test_loader, writer, epoch)

if __name__ == '__main__':
    main()
