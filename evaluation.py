#Evaluate performance
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np

from tqdm import tqdm
from utils import multiclass_dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, last_epochs=False, num_classes=1):
    net.eval()
    criterion = nn.CrossEntropyLoss()

    # Evaluation
    total_loss = 0
    dice_total, iou_total, acc_total = 0, 0, 0
    precision_total, recall_total, f1_total = 0, 0, 0

    # Confusion matrix
    y_true_all, y_pred_all = [], []

    # Iterate over the validation set
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # Move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                # Predict the mask
                mask_pred = net(image)
                mask_pred = TF.resize(mask_pred, (mask_true.shape[1], mask_true.shape[2]),
                                      interpolation=TF.InterpolationMode.NEAREST)
                prediction = mask_pred.argmax(dim=1)

                if last_epochs:
                    y_true_all.append(mask_true.cpu().numpy())
                    y_pred_all.append(mask_pred.argmax(dim=1).cpu().numpy())

                # Compute validation loss
                loss = criterion(mask_pred, mask_true)
                total_loss += loss.item()

                # Compute the Dice score, ignoring background
                dice_total += multiclass_dice_coeff(prediction, mask_true).item()

                # Compute IoU
                intersection = (prediction & mask_true).float().sum()
                union = (prediction | mask_true).float().sum()
                iou = (intersection + 1e-6) / (union + 1e-6)
                iou_total += iou.item()

                # Compute accuracy
                acc_total += (prediction == mask_true).float().mean().item()

                # Compute Precision, Recall and F1-Score
                tp = ((prediction == mask_true) & (mask_true > 0)).sum().float()
                fp = ((prediction != mask_true) & (prediction > 0)).sum().float()
                fn = ((prediction != mask_true) & (mask_true > 0)).sum().float()

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)

                precision_total += precision.item()
                recall_total += recall.item()
                f1_total += f1.item()

    n = len(dataloader)
    return total_loss / n, dice_total / n, iou_total / n, acc_total / n, precision_total / n, recall_total / n, f1_total / n, np.array(
        y_true_all), np.array(y_pred_all)