#Predict label shape in images
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils import BasicDataset
from model import UNet
from utils import plot_img_and_mask

#Create mask from images
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    #net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

class Train_Args:
    def __init__(self):
        self.model = None
        self.input = 'image01.jpg'
        self.output = None
        self.viz = None
        self.no_save = True
        self.mask_threshold = 0.2
        self.scale = 0.5
        self.bilinear = True
        self.classes = 5

def get_args():
    '''parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')'''

    def ask(prompt, cast_type, default_value):
        user_input = input(f"{prompt} (default: {default_value}): ").strip()
        if user_input == "":
            return default_value
        try:
            return cast_type(user_input)
        except ValueError:
            print("This value is not ")
            return default_value

    #Initialize variables
    args = Train_Args()

    print("<*> Please supply the following arguments:")
    args.model = input("Model path:").strip()
    if args.model == "":
        args.model = "/kaggle/input/dataset-unet/checkpoints/checkpoint_epoch50.pth"
    args.input = input("Input path:").strip()
    if args.input == "":
        args.input = '/kaggle/input/dataset-unet/Dataset_UNet/test/Image13.jpg'
    args.output = input("Output path:").strip()
    if args.output == "":
        args.output = '/kaggle/working/Output13.jpg'
    args.viz = ask("Do you want to visualize image?", bool, True)

    save = ask("Do you want to save the output masks?", bool, True)
    if save:
        args.no_save = False
    else:
        args.no_save = True

    return args

#Save mask
def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    #out_files = get_output_filenames(args)
    out_files = args.output

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'<*> Loading model: {args.model}')
    print(f'<*> Using device: {device}')

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    print('>>> Model loaded!')

    logging.info('Model loaded!')

    print(f'\n<*> Predicting image {in_files} ..........')
    logging.info(f'Predicting image {in_files} ...')
    img = Image.open(in_files)

    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=args.scale,
                        out_threshold=args.mask_threshold,
                        device=device)

    if not args.no_save:
        result = mask_to_image(mask, mask_values)
        result.save(out_files)
        print(f'\n<*> Mask saved to {out_files} ..........')
        logging.info(f'Mask saved to {out_files}')

    if args.viz:
        print(f'\n<*> Visualizing results for image {in_files}, close to continue..........')
        logging.info(f'Visualizing results for image {in_files}, close to continue...')
        plot_img_and_mask(img, mask)