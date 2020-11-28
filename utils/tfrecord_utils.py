import os
import re
import glob

import cv2
import tensorflow as tf
from scipy.io import loadmat


def image_example(image_path, output_shape):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = loadmat(os.path.splitext(image_path)[0] + '.mat')['annPoints']
        label_x = label[:, 0] / image.shape[1]
        label_y = label[:, 1] / image.shape[0]
        image = cv2.resize(image, dsize=output_shape)
        image_shape = image.shape
        image_string = cv2.imencode('.png', image)[1].tostring()
    except IndexError as e:
        print(e, 'Indexing occurred')

    feature = {
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[0]])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[1]])),
        'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[2]])),
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=label_x)),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=label_y)),
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
    train_path = "/home/ferhat/PycharmProjects/crowd-density-dl/data/Train"
    test_path = "/home/ferhat/PycharmProjects/crowd-density-dl/data/Test"
    train_record_file = '/home/ferhat/PycharmProjects/tf-faster-rcnn/data/train_images.tfrecords'
    test_record_file = '/home/ferhat/PycharmProjects/tf-faster-rcnn/data/test_images.tfrecords'

    for path, record_file in zip([train_path, test_path], [train_record_file, test_record_file]):
        fn_list = glob.glob(os.path.join(path, '*.*'))
        with tf.io.TFRecordWriter(record_file) as writer:
            for file in fn_list:
                if re.search(r'.*jpg$|.*jpeg$|.*png$', file.lower()):
                    try:
                        tf_example = image_example(file, args.output_shape)
                    except Exception as e:
                        print('Error creating tf.Example', e)
                        continue
                    writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    # main()
    raw_image_dataset = tf.data.TFRecordDataset('/home/ferhat/PycharmProjects/tf-faster-rcnn/data/train_images.tfrecords')
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.VarLenFeature(tf.float32),
        'y': tf.io.VarLenFeature(tf.float32),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:
        image_raw = tf.io.decode_image(image_features['image_raw']).numpy()
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        cv2.imshow('im', image_raw)
        key = cv2.waitKey(0)
        if key == ord('n'):
            continue
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break
