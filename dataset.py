import tensorflow as tf
import numpy as np
import cv2

def DateSet(file_list, args, debug=False):
    file_list, landmarks, attributes,euler_angles = gen_data(file_list)
    if debug:
        n = args.batch_size * 10
        file_list = file_list[:n]
        landmarks = landmarks[:n]
        attributes = attributes[:n]
        euler_angles=euler_angles[:n]
    dataset = tf.data.Dataset.from_tensor_slices((file_list, landmarks, attributes,euler_angles))

    def _parse_data(filename, landmarks, attributes,euler_angles):
        # filename, landmarks, attributes = data
        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=args.image_channels)
        # print(image.get_shape())
        # image.set_shape((args.image_size, args.image_size, args.image_channels))
        image = tf.image.resize_images(image, (args.image_size, args.image_size), method=0)
        image = tf.cast(image, tf.float32)

        image = image / 256.0
        return (image, landmarks, attributes,euler_angles)

    dataset = dataset.map(_parse_data)
    dataset = dataset.shuffle(buffer_size=10000)
    return dataset, len(file_list)

def gen_data(file_list):
    with open(file_list,'r') as f:
        lines = f.readlines()
    filenames, landmarks,attributes,euler_angles = [], [], [],[]
    for line in lines:
        line = line.strip().split()
        path = line[0]
        landmark = line[1:197]
        attribute = line[197:203]
        euler_angle = line[203:206]

        landmark = np.asarray(landmark, dtype=np.float32)
        attribute = np.asarray(attribute, dtype=np.int32)
        euler_angle = np.asarray(euler_angle,dtype=np.float32)
        filenames.append(path)
        landmarks.append(landmark)
        attributes.append(attribute)
        euler_angles.append(euler_angle)
        
    filenames = np.asarray(filenames, dtype=np.str)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.int32)
    euler_angles = np.asarray(euler_angles,dtype=np.float32)
    return (filenames, landmarks, attributes,euler_angles)


if __name__ == '__main__':
    file_list = 'data/train_data/list.txt'
    filenames, landmarks, attributes = gen_data(file_list)
    for i in range(len(filenames)):
        filename = filenames[i]
        landmark = landmarks[i]
        attribute = attributes[i]
        print(attribute)
        img = cv2.imread(filename)
        h,w,_ = img.shape
        landmark = landmark.reshape(-1,2)*[h,w]
        for (x,y) in landmark.astype(np.int32):
            cv2.circle(img, (x,y),1,(0,0,255))
        cv2.imshow('0', img)
        cv2.waitKey(0)
