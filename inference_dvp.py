import argparse
import os
import numpy as np
import torch

from models.pfld import PFLDInference
from inference import detect_images_landmarks


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pfld_model_path', "-pm", type=str, default="./checkpoint/snapshot/checkpoint.pth.tar")
    parser.add_argument("--model_dir", "-m", type=str, required=True)
    parser.add_argument('--train_3d_with_align', action='store_true', default=False)
    parser.add_argument('--is_plot_landmark_debug', action='store_true', default=False)

    return parser.parse_args()


def main():
    args = arg_parser()

    pfld_model_path = args.pfld_model_path
    model_dir = args.model_dir
    train_3d_with_align = args.train_3d_with_align
    is_plot_landmark_debug = args.is_plot_landmark_debug

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(pfld_model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])

    output_dir = os.path.join(model_dir, "debug_hrnet_68_lm")
    os.makedirs(output_dir, exist_ok=True)

    if train_3d_with_align:
        images_dir = os.path.join(model_dir, 'align')
    else:
        images_dir = os.path.join(model_dir, 'crop')

    landmark_list = detect_images_landmarks(pfld_backbone, images_dir,
                                            is_plot_landmark=is_plot_landmark_debug, device=device)
    np.save(os.path.join(model_dir, 'landmark_pfld.npy'), landmark_list)


if __name__ == '__main__':
    main()



