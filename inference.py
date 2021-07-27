import os
import cv2
import torch
from tqdm import tqdm


def inference_single_image_98_lm(pfld_backbone, image, device='cuda'):
    pfld_backbone.eval()

    height, width = image.shape[0], image.shape[1]
    with torch.no_grad():
        img = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
        img = img.to(device)

        _, landmarks = pfld_backbone(img)
        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.reshape(-1, 2) * [width, height]
    # print("landmark:", landmarks)
    # print("landmark_pre_max:", landmarks.max())
    # print("landmark_pre_min:", landmarks.min())

    return landmarks


def plot_landmarks(landmarks, image, save_result_path="result.jpg"):
    for idx, point in enumerate(landmarks):
        image = cv2.circle(image, (int(point[0]), int(point[1])), radius=1, color=(255, 0, 0), thickness=1)
    cv2.imwrite(save_result_path, image)


def detect_images_landmarks(pfld_backbone, image_dir_name, is_plot_landmark=False, device='cuda'):
    pfld_backbone = pfld_backbone.to(device)

    if is_plot_landmark:
        output_dir = os.path.join(image_dir_name, "..", "debug_pfld_98_lm")
        os.makedirs(output_dir, exist_ok=True)

    landmark_list = []
    image_names_list = sorted(os.listdir(image_dir_name))
    for image_name in tqdm(image_names_list):
        image_path = os.path.join(image_dir_name, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (112, 112))
        landmarks = inference_single_image_98_lm(pfld_backbone, image, device)

        if is_plot_landmark:
            save_result_path = os.path.join(output_dir, image_name)
            plot_landmarks(landmarks, image, save_result_path)
        landmark_list.append(landmarks)

    if is_plot_landmark:
        print(f"Images were saved at {output_dir}.")
    return landmark_list
