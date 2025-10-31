#Train model
import argparse
import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import optim
from torch.utils.data import DataLoader, random_split

from pathlib import Path
from tqdm import tqdm
import wandb
from glob import glob

from utils import BasicDataset, CarvanaDataset
from utils import dice_loss
from evaluation import evaluate
from utils import plot_confusion_matrix
from model import UNet

#Import library for statistic
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

#Direct to the folder
dir_img = "/kaggle/input/dataset-unet/images"
dir_mask = "/kaggle/input/dataset-unet/masks"
dir_checkpoint = "/kaggle/working/"

def compute_class_weights(mask_dir, n_classes, mask_values):
    counts = np.zeros(n_classes, dtype=np.float64)
    mask_paths = glob(mask_dir + '/*.png')

    for mask_path in tqdm(mask_paths, desc='<*> Calculating class frequencies ..........'):
        mask = np.array(Image.open(mask_path))
        for i, v in enumerate(mask_values):
            counts[i] += np.sum(mask == v)

    total = np.sum(counts)
    freq = counts / total

    #Compute class weights
    weights = 1.0 / (np.log(1.02 + freq))
    weights = torch.tensor(weights, dtype=torch.float32)
    print("<*> Class frequencies:", freq)
    print("<*> Computed class weights:", weights)
    print()
    return weights

def train_model(
        model,
        device,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-5,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    #1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale, augment=False)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale, augment=False)

    #2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    #3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    #(Initialize logging)
    '''experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )'''

    logging.info(f'''<*> Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    print(f'''\n<*> Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    #4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    class_weights = compute_class_weights(dir_mask, model.n_classes, dataset.mask_values)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    global_step = 0
    save_dir = "/content/drive/MyDrive/Colab Notebooks/U-Net+VGG16/metrics"
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    dice_scores = []
    iou_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    #(Data augmentation):
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    #5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        last_epochs = (epoch == epochs)
        with tqdm(total=n_train, desc=f'<*> Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    masks_pred = TF.resize(masks_pred, (true_masks.shape[1], true_masks.shape[2]), interpolation=TF.InterpolationMode.NEAREST)

                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                '''experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })'''
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                #Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        '''histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())'''

                        val_loss, dice_score, iou_score, accuracy, precision_score, recall_score, f1_score, y_true_all, y_pred_all = evaluate(model, val_loader, device, amp, last_epochs, model.n_classes)
                        scheduler.step(dice_score)
                        print("\n<*> Validation loss:", val_loss)
                        print("<*> Validation Dice score:", dice_score)
                        print("<*> IoU Score:", iou_score)
                        print("<*> Accuracy: {}%".format(accuracy*100))
                        print("<*> Precicion: {}".format(precision_score*100))
                        print("<*> Recall: {}".format(recall_score*100))
                        print("<*> F1_Score: {}".format(f1_score*100))

                        '''logging.info('<*> Validation Dice score: {}'.format(dice_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': dice_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass'''

        #Save in list
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)
        dice_scores.append(dice_score)
        iou_scores.append(iou_score)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        f1_scores.append(f1_score)

        if epoch % 25 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint + '/checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'<*> Checkpoint {epoch} saved!')
            print(f'<*> Checkpoint {epoch} saved!\n')

    #6. Export evaluate metrics in charts
    save_dir = "/kaggle/working/"
    os.makedirs(save_dir, exist_ok=True)

    epochs_range = range(1, epochs + 1)

    #Train and Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs_range, val_losses, label='Validation Loss', color='orange')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Train_Val_Loss.png'))
    plt.show()

    #Dice and IoU Score
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, dice_scores, label='Dice Score', color='blue')
    plt.plot(epochs_range, iou_scores, label='IoU Score', color='orange')
    plt.title('Dice vs IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Dice_IoU_Score.png'))
    plt.show()

    #Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, accuracy_scores, label='Accuracy', color='blue')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Accuracy.png'))
    plt.show()

    #Precision, Recall, F1-Score
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, precision_scores, label='Precision', color='blue')
    plt.plot(epochs_range, recall_scores, label='Recall', color='orange')
    plt.plot(epochs_range, f1_scores, label='F1-Score', color='black')
    plt.title('Precision, Recall, F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Precision_Recall_F1.png'))
    plt.show()

    #Confusion Matrix
    class_names = ["Background", "Xe đạp", "Xe máy", "Xe bốn bánh", "Phương tiện khác"]
    save_path = f"{save_dir}/Confusion_Matrix.png"
    plot_confusion_matrix(y_true_all, y_pred_all, class_names, save_path)
    print(f"<*> Confusion matrix saved to {save_path}")

class Args:
    def __init__(self):
        self.epochs = None
        self.batch_size = None
        self.lr = None
        self.load = None
        self.scale = 0.5
        self.val = 10
        self.amp = False
        self.bilinear = False
        self.classes = 5

def get_args():
    '''parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')'''

    #Initialize args
    args = Args()

    print("<*> Please supply the following arguments:")
    def ask(prompt, cast_type, default_value):
        user_input = input(f"{prompt} (default: {default_value}): ").strip()
        if user_input == "":
            return default_value
        try:
            return cast_type(user_input)
        except ValueError:
            print("This value is not ")
            return default_value

    #Get value from keyboard
    args.epochs = ask("Number of epoch:", int, 50)
    args.batch_size = ask("Batch size:", int, 16)
    args.lr = ask("Learning rate:", float, 1e-4)
    args.load = input("Model path:").strip()
    if args.load == "":
        args.load = False
    args.scale = ask("Down image scale:", float, 0.5)
    args.val = ask("Image percentage for validation:", float, 20.0)
    args.amp = input("Use mixed precision AMP? (y/n):").strip().lower() == "y"
    args.bilinear = input("Use bilinear upsampling? (y/n):").strip().lower() == "y"
    args.classes = ask("Number of classes:", int, 5)

    return args

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    print(f'<*> Using device: {device}')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'<*> Network:\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    print(f'<*> Network:\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )