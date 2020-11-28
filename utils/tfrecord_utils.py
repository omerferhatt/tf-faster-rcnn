import os
import re
import glob
import argparse

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
        feature = {
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[0]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[1]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[2]])),
            'total_human': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(label_x)])),
            'x': tf.train.Feature(float_list=tf.train.FloatList(value=label_x)),
            'y': tf.train.Feature(float_list=tf.train.FloatList(value=label_y)),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string]))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    except Exception as e:
        print(e)


def write_record(train_path, test_path, train_record_file, test_record_file):
    for path, record_file in zip([train_path, test_path], [train_record_file, test_record_file]):
        fn_list = glob.glob(os.path.join(path, '*.*'))
        with tf.io.TFRecordWriter(record_file) as writer:
            for file in fn_list:
                if re.search(r'.*jpg$|.*jpeg$|.*png$', file.lower()):
                    try:
                        tf_example = image_example(file, arg.output_shape)
                    except Exception as e:
                        print('Error creating tf.Example', e)
                        continue
                    writer.write(tf_example.SerializeToString())


def read_record(tf_record_path='/home/ferhat/PycharmProjects/tf-faster-rcnn/data/train_images.tfrecords'):
    raw_image_dataset = tf.data.TFRecordDataset(tf_record_path)
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

    return raw_image_dataset.map(_parse_image_function)


def parser():
    pars = argparse.ArgumentParser()
    pars.add_argument('--train-data-path', type=str,
                      default='/home/ferhat/PycharmProjects/crowd-density-dl/data/Train')
    pars.add_argument('--test-data-path', type=str,
                      default='/home/ferhat/PycharmProjects/crowd-density-dl/data/Test')
    pars.add_argument('--train-record', type=str,
                      default='/home/ferhat/PycharmProjects/tf-faster-rcnn/data/train_images.tfrecords')
    pars.add_argument('--test-record', type=str,
                      default='/home/ferhat/PycharmProjects/tf-faster-rcnn/data/test_images.tfrecords')
    return pars.parse_args()


def test_record(dataset):
    for image_features in dataset:
        image_raw = tf.io.decode_image(image_features['image_raw']).numpy()
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        cv2.imshow('im', image_raw)
        key = cv2.waitKey(0)
        if key == ord('n'):
            continue
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    arg = parser()
    parsed_image_dataset = read_record()
    test_record(parsed_image_dataset)
