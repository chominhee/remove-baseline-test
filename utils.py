
import tensorflow as tf  # I used Tensorflow-GPU 2.0.
import os, sys
import argparse
import logging

import pickle as pkl


# set logger ###################################################################
logger = logging.getLogger(__name__)
hLoggerFile = logging.StreamHandler(sys.stdout)
hLoggerFile.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)s %(message)s'))
logger.addHandler(hLoggerFile)
logger.setLevel(os.environ.get("LOG_LEVEL", "ERROR"))


# end of set logger ############################################################


def input_from_file(in_dir: str, data_name: list):
    """ output to files into given output directory """

    print(f"input file: {data_name}\n" )
    try:
        # read datas from each pickle files
        data_list = []
        data_num = len(data_name)
        for i in range(data_num):
            ipath = os.path.join(in_dir, data_name[i] + '.pkl')
            with open(ipath, 'rb') as fd:
                data_list.append(pkl.load(fd))
        return data_list

    except Exception as e:
        logger.error("exception for read data into file: {}\n".format(e))
        return None 


def output_to_file(out_dir: str, data_name: list, data: list):
    """ output to files into given output directory """
    print(f"input file: {data_name}\n" )
    try:
        # write datas to each pickle files
        data_num = len(data_name)
        for i in range(data_num):
            opath = os.path.join(out_dir, data_name[i] + '.pkl')
            with open(opath, 'wb+') as fd:
                pkl.dump(data[i], fd)
        return True

    except Exception as e:
        logger.error("exception for write data into file: {}\n".format(e))
        return False


def save_model(out_dir: str, model_name: str, model):
    try:
        # write datas to each pickle files
        spath = os.path.join(out_dir, model_name)
        model.save(spath)
        return True

    except Exception as e:
        logger.error("exception for write data into file: {}\n".format(e))
        return False


def gradient(model, inputs, labels):
    with tf.GradientTape() as tape:
        y_hat, _ = model(inputs)
        loss = tf.keras.losses.binary_crossentropy(labels, y_hat)

    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad