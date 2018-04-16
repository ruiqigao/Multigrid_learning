import tensorflow as tf
import os

from model import Multigrid
from utils import pp

flags = tf.app.flags
flags.DEFINE_boolean('prefetch', True, 'True for prefetch images')
flags.DEFINE_string('dataset_name', 'rock', 'Folder name which stored in ./data/dataset_name')
flags.DEFINE_string('input_pattern', '*.jpg', 'Glob pattern of filename of input images [*]')
flags.DEFINE_string('sample_dir', 'sample', 'Glob pattern of filename of input images [*]')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'Glob pattern of filename of input images [*]')
flags.DEFINE_integer('T', 90, 'Number of Langevin iterations')
flags.DEFINE_integer('batch_size', 100, 'Batch size for training')
flags.DEFINE_integer('epoch', 700, 'Number of epoch for training')
flags.DEFINE_integer('image_size', 64, 'image size of training images')
flags.DEFINE_integer('num_threads', 2, 'threads for read images')
flags.DEFINE_integer('num_gpus', 1, 'number of gpu used in calculation')
flags.DEFINE_integer('read_len', 100, 'number of batches per reading')
flags.DEFINE_float('delta', 0.3, 'Langevin step size')
flags.DEFINE_float('learning_rate', 0.3, 'Learning rate')
flags.DEFINE_float('beta1', 0.5, 'Momentum')
flags.DEFINE_float('weight_decay', 0.0001, 'weight_decay')
flags.DEFINE_float('ref_sig', 50, 'Standard deviation for reference gaussian distribution')
flags.DEFINE_float('clip_grad', 1.0, 'clipped maximum gradient for update')
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        models = Multigrid(sess, FLAGS)

    models.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
