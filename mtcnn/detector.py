import math
import numpy as np
from PIL import Image
import torch

from model import PNet, RNet, ONet
from utils import nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess, show_bboxes

def detect_faces(image, min_face_size=40.0, thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    pnet, rnet, onet= PNet(), RNet(), ONet()
    onet.eval()

    width, height = image.size
    min_length = min(height, width)
    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    scales = []
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1
    bounding_boxes = []
    for s in scales:    # run P-Net on different scales
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2
    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    img_boxes = torch.FloatTensor(img_boxes)
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    probs = output[1].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3
    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0: 
        return [], []
    img_boxes = torch.FloatTensor(img_boxes)
    output = onet(img_boxes)
    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks

def run_first_stage(image, net, scale, threshold):
    """ 
        Run P-Net, generate bounding boxes, and do NMS.
    """
    width, height = image.size
    sw, sh = math.ceil(width*scale), math.ceil(height*scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, 'float32')
    img = torch.FloatTensor(_preprocess(img))

    output = net(img)
    probs = output[1].data.numpy()[0, 1, :, :]
    offsets = output[0].data.numpy()

    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None

    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    """
       Generate bounding boxes at places where there is probably a face.
    """
    stride = 2
    cell_size = 12

    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images, so we need to rescale bounding boxes back
    bounding_boxes = np.vstack([
        np.round((stride*inds[1] + 1.0)/scale),
        np.round((stride*inds[0] + 1.0)/scale),
        np.round((stride*inds[1] + 1.0 + cell_size)/scale),
        np.round((stride*inds[0] + 1.0 + cell_size)/scale),
        score, offsets
    ])

    return bounding_boxes.T

if __name__ == '__main__':
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret: break
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        bounding_boxes, landmarks = detect_faces(img)
        image = show_bboxes(img, bounding_boxes, landmarks)
        image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
        cv2.imshow('0', image)
        if cv2.waitKey(10) == 27:
            break