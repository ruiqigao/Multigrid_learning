# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *


class Multigrid(object):
    def __init__(self, sess, flags, scale_list=[1, 4, 16, 64]):
        self.sess = sess
        self.batch_size = flags.batch_size
        self.weight_decay = flags.weight_decay
        self.scale_list = scale_list
        self.images = {}
        self.images_inv = {}
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.original_images = {}
        self.synthesized_images = {}
        self.m_original = {}
        self.m_synthesized = {}
        self.vars = {}
        self.sample_loss = {}
        self.train_loss = {}
        self.recon_loss = {}
        self.train_op = {}
        self.sampling_op = {}
        for to_sz in scale_list:
            self.images[to_sz] = []
            self.images_inv[to_sz] = []

        # create image placeholder for down_sampling
        from_sz = scale_list[-1]
        self.images[from_sz] = tf.placeholder(tf.float32, shape=[None, from_sz, from_sz, 3])
        for to_sz in scale_list[0:-1]:
            self.images[to_sz] = self.build_Q(self.images[from_sz], from_sz, to_sz)
        # build image placeholder for up_sampling
        from_sz = scale_list[0]
        self.images_inv[from_sz] = tf.placeholder(tf.float32, shape=[None, from_sz, from_sz, 3])
        for to_sz in scale_list[1:]:
            self.images_inv[to_sz] = self.build_Q_inv(self.images_inv[from_sz], from_sz, to_sz)
            from_sz = to_sz

        if flags.prefetch:
            files = glob(os.path.join('./data', flags.dataset_name, flags.input_pattern))
            self.data = par_imread(files, flags.image_size, flags.num_threads)
        else:
            self.data = glob(os.path.join('./data', flags.dataset_name, flags.input_pattern))
        self.build_model(flags)

    def build_Q(self, input, from_size, to_size, reuse=False):
        assert from_size % to_size == 0, "Setup error: the from_size({}) should be divisible by to_size({})".\
            format(from_size, to_size)
        var_scope = 'transfer_{}_{}'.format(from_size, to_size)
        filter_size = from_size / to_size
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                scope.reuse_variables()

            ratio_value = np.float32(1.0 / filter_size / filter_size)
            temp_w = np.zeros((filter_size, filter_size, 3, 3), np.float32)
            temp_w[:, :, 0, 0] = ratio_value
            temp_w[:, :, 1, 1] = ratio_value
            temp_w[:, :, 2, 2] = ratio_value
            Q = tf.Variable(temp_w, name=var_scope + '_Q', trainable=False)

            # data format：[batch, height, width, channels]
            down_sampled = tf.nn.conv2d(input, Q, [1, filter_size, filter_size, 1], padding='SAME')

            return down_sampled

    def build_Q_inv(self, input, from_size, to_size, reuse = False):
        assert to_size % from_size == 0, "Setup error: the to_size({}) should be divisible by from_size({})".\
            format(to_size, from_size)
        var_scope = 'transfer_inv_{}_{}'.format(from_size, to_size)
        filter_size = to_size / from_size
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                scope.reuse_variables()

            temp_w_inv = np.zeros((filter_size, filter_size, 3, 3), np.float32)
            temp_w_inv[:, :, 0, 0] = 1
            temp_w_inv[:, :, 1, 1] = 1
            temp_w_inv[:, :, 2, 2] = 1
            Q_inv = tf.Variable(temp_w_inv, name=var_scope + '_Qinv', trainable=False)
            batch_size = tf.shape(input)[0]
            deconv_shape = [batch_size, to_size, to_size, 3]
            up_sampled = tf.nn.conv2d_transpose(input, Q_inv, output_shape=deconv_shape,
                                                strides=[1, filter_size, filter_size, 1])

            return up_sampled

    def descriptor_warpper(self, inputs, im_sz, is_training=True, reuse=False):
        if im_sz == 64:
            return self.descriptor64(inputs, is_training, reuse)
        elif im_sz == 16:
            return self.descriptor16(inputs, is_training, reuse)
        elif im_sz == 4:
            return self.descriptor4(inputs, is_training, reuse)
        else:
            print('Error!! unsupported model version {}'.format(im_sz))
            exit()

    def descriptor64(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope('des64') as scope:
            if reuse:
                scope.reuse_variables()
            kernel_reg = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
            kernel_init = tf.contrib.layers.xavier_initializer(True)
            # layer 1 5x5, stride = 2, pad = 2, n_out = 96
            h0 = conv_layer(inputs, 64, 5, 2, is_training, kernel_reg, kernel_init, 0, 0.2)
            # layer 2 3x3, stride = 2, pad = 2, n_out = 256
            h1 = conv_layer(h0, 128, 5, 2, is_training, kernel_reg, kernel_init, 1, 0.2)
            # layer 3 3x3, stride = 1, pad = 2, n_out = 256
            h2 = conv_layer(h1, 256, 5, 2, is_training, kernel_reg, kernel_init, 2, 0.2)
            # layer 4 5*5 stride = 2, pad = 2, out = 512
            h3 = conv_layer(h2, 512, 5, 2, is_training, kernel_reg, kernel_init, 3, 0.2)
            # layer 4 fully connected, out = 1
            num_out = int(h3.shape[1] * h3.shape[2] * h3.shape[3])
            h4 = tf.layers.dense(tf.reshape(h3, [-1, num_out], name='reshape'), 1,
                                 kernel_regularizer=kernel_reg, kernel_initializer=kernel_init, name='fc')
            return h4

    def descriptor16(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope('des16') as scope:
            if reuse:
                scope.reuse_variables()

            kernel_reg = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
            kernel_init = tf.contrib.layers.xavier_initializer(True)
            # layer 1 5x5, stride = 2, pad = 2, n_out = 96
            h0 = conv_layer(inputs, 96, 5, 2, is_training, kernel_reg, kernel_init, 0, 0.2)
            # layer 2 3x3, stride = 1, pad = 2, n_out = 256
            h1 = conv_layer(h0, 128, 3, 1, is_training, kernel_reg, kernel_init, 1, 0.2)
            # layer 3 3x3, stride = 1, pad = 2, n_out = 256
            h2 = conv_layer(h1, 256, 3, 1, is_training, kernel_reg, kernel_init, 2, 0.2)
            # layer 4 3x3, stride = 1, pad = 2, n_out = 256
            h3 = conv_layer(h2, 512, 3, 1, is_training, kernel_reg, kernel_init, 3, 0.2)
            # layer 5 fully connected, out = 1
            num_out = int(h3.shape[1] * h3.shape[2] * h3.shape[3])
            h4 = tf.layers.dense(tf.reshape(h3, [-1, num_out], name='reshape'), 1,
                                 kernel_regularizer=kernel_reg, kernel_initializer=kernel_init, name='fc')

            return h4

    def descriptor4(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope('des4') as scope:
            if reuse:
                scope.reuse_variables()

            kernel_reg = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
            kernel_init = tf.contrib.layers.xavier_initializer(True)
            # layer 1 5x5, stride = 2, pad = 2, n_out = 96
            h0 = conv_layer(inputs, 96, 5, 2, is_training, kernel_reg, kernel_init, 0, 0.2)
            # layer 2 3x3, stride = 1, pad = 2, n_out = 256
            h1 = conv_layer(h0, 128, 3, 1, is_training, kernel_reg, kernel_init, 1, 0.2)
            # layer 3 3x3, stride = 1, pad = 2, n_out = 256
            h2 = conv_layer(h1, 256, 3, 1, is_training, kernel_reg, kernel_init, 2, 0.2)
            # layer 4 fully connected, out = 1
            num_out = int(h2.shape[1] * h2.shape[2] * h2.shape[3])
            h3 = tf.layers.dense(tf.reshape(h2, [-1, num_out], name='reshape'), 1,
                                 kernel_regularizer=kernel_reg, kernel_initializer=kernel_init, name='fc')

            return h3

    def Langevin_sampling(self, samples, to_sz, flags):
        def cond(i, samples):
            return tf.less(i, flags.T)

        def body(i, samples):
            syn_res = self.descriptor_warpper(samples, to_sz, is_training=True, reuse=True)
            grad = tf.gradients(syn_res, samples, name='grad_des')[0]
            samples = samples + 0.5 * flags.delta * flags.delta * grad
            samples = tf.clip_by_value(samples, 0, 255)
            return tf.add(i, 1), samples

        i = tf.constant(0)
        i, samples = tf.while_loop(cond, body, [i, samples])

        return samples

    def build_model(self, flags):
        m_optim = {}
        grads_and_vars = {}

        for im_sz in self.scale_list[1:]:
            # define optimizer and training option
            m_optim[im_sz] = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1)

            self.original_images[im_sz] = tf.placeholder(tf.float32, shape=[None, im_sz, im_sz, 3])
            self.synthesized_images[im_sz] = tf.placeholder(tf.float32, shape=[None, im_sz, im_sz, 3])
            self.m_original[im_sz] = self.descriptor_warpper(self.original_images[im_sz], im_sz, self.phase)
            self.m_synthesized[im_sz] = self.descriptor_warpper(self.synthesized_images[im_sz], im_sz, self.phase, True)

        t_vars = tf.trainable_variables()

        for im_sz in self.scale_list[1:]:
            self.vars[im_sz] = [var for var in t_vars if 'des{}'.format(im_sz) in var.name]

            self.sample_loss[im_sz] = tf.reduce_sum(self.m_synthesized[im_sz])
            # To maximize the log-likelihood, w-8.13556e-05e minimize the negative log-likelihood:
            # grad = grad( tf.reduce_sum((self.m_64_synthesized) - tf.reduce_sum((self.m_64_original) )
            self.train_loss[im_sz] = tf.subtract(tf.reduce_mean(
                self.m_synthesized[im_sz]), tf.reduce_mean(self.m_original[im_sz]))

            self.recon_loss[im_sz] = tf.reduce_mean(
                tf.abs(tf.subtract(self.original_images[im_sz], self.synthesized_images[im_sz])))

            # define gradient update and clipping policy
            grads_and_vars[im_sz] = m_optim[im_sz].compute_gradients(self.train_loss[im_sz], var_list=self.vars[im_sz])

            # do the summary
            initial_flag = 0
            for grad, var in grads_and_vars[im_sz]:
                if 'kernel' in var.name:
                    if initial_flag == 0:
                        tmpgrad = tf.reshape(grad, [-1])
                        initial_flag = 1
                    else:
                        tmpgrad = tf.concat([tmpgrad, tf.reshape(grad, [-1])], 0)

            tf.summary.scalar('kernel_des{}_maxgrad'.format(im_sz), tf.reduce_max(tf.abs(tmpgrad)))
            tf.summary.scalar('kernel_des{}_meangrad'.format(im_sz), tf.reduce_mean(tf.abs(tmpgrad)))
            tf.summary.scalar('kernel_des{}_normgrad'.format(im_sz), tf.norm(tmpgrad))

            self.train_op[im_sz] = m_optim[im_sz].apply_gradients(grads_and_vars[im_sz])
            self.sampling_op[im_sz] = self.Langevin_sampling(self.synthesized_images[im_sz], im_sz, flags)

    def train(self, flags):
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        self.mysummary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

        self.sess.graph.finalize()
        counter = 1
        start_time = time.time()

        batch_idxs = int(math.ceil(float(len(self.data)) / self.batch_size))
        # small_batchs = int(math.ceil(float(self.batch_size) / flags.num_gpus))
        burst_len = (self.batch_size * flags.read_len)

        could_load, checkpoint_counter = self.load(flags)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            start_epoch = int(math.floor(float(counter - 1) / batch_idxs))
            start_idx = np.mod((counter - 1), batch_idxs)
            self.global_step = (counter - 1) * (len(self.scale_list) - 1)

            if not flags.prefetch:
                start_idx = int(math.floor(float(start_idx) / burst_len)) * burst_len
                end_idx = min(start_idx + burst_len, len(self.data))
                files = self.data[start_idx: end_idx]
                tmp_list = par_imread(files, flags.image_size, flags.num_threads)

        else:
            print(" [!] Load failed...")
            start_idx = 0
            start_epoch = 0

        for epoch in xrange(start_epoch, flags.epoch):
            for idx_batch in xrange(start_idx, batch_idxs):
                if flags.prefetch is False and np.mod(idx_batch, flags.read_len) == 0:
                    start_idx = idx_batch * self.batch_size
                    end_idx = min(start_idx + burst_len, len(self.data))
                    files = self.data[start_idx: end_idx]
                    tmp_list = par_imread(files, flags.image_size, flags.num_threads)

                start_idx = idx_batch * self.batch_size
                end_idx = min((idx_batch+1) * self.batch_size, len(self.data))
                if flags.prefetch:
                    batch_images = np.array(self.data[start_idx: end_idx]).astype(np.float32)
                else:
                    start_idx = np.mod(start_idx, burst_len)
                    end_idx = np.mod(end_idx, burst_len)
                    if end_idx == 0:
                        end_idx = burst_len
                    batch_images = np.array(tmp_list[start_idx: end_idx]).astype(np.float32)

                # generate initial samples
                to_sz = self.scale_list[0]
                from_sz = self.scale_list[-1]

                tmp_feed_dict = {}
                tmp_feed_dict[self.images[from_sz]] = batch_images

                samples = {}
                train_images = {}
                # downsample to 1x1 images
                samples[to_sz] = self.sess.run(self.images[to_sz],
                                               feed_dict={self.images[self.scale_list[-1]]: batch_images})
                from_sz = to_sz
                for to_sz in self.scale_list[1:]:
                    # training images in this scale
                    if to_sz != self.scale_list[-1]:
                        tmp = self.sess.run(self.images[to_sz],
                                            feed_dict={self.images[self.scale_list[-1]]: batch_images})
                        train_images[to_sz] = np.array(tmp).astype(np.float32)
                    else:
                        train_images[to_sz] = batch_images

                    # up sampling
                    samples[to_sz] = self.sess.run(self.images_inv[to_sz],
                                                   feed_dict={self.images_inv[from_sz]: samples[from_sz]})

                    # run Langevin sampling on images
                    samples[to_sz] = self.sess.run(self.sampling_op[to_sz],
                                                   feed_dict={self.synthesized_images[to_sz]: samples[to_sz], self.phase: True})

                    # Compute reconstruction error
                    tmp_feed_dict = {}
                    tmp_feed_dict[self.phase] = True
                    tmp_feed_dict[self.original_images[to_sz]] = train_images[to_sz]
                    tmp_feed_dict[self.synthesized_images[to_sz]] = samples[to_sz]

                    [err_list, err_list2] = self.sess.run([self.train_loss[to_sz], self.recon_loss[to_sz],
                                                           self.train_op[to_sz]], feed_dict=tmp_feed_dict)[:2]
                    err = np.mean(err_list)
                    err2 = np.mean(err_list2)

                    if np.mod(counter, 1000) == 1:
                        save_images(samples[to_sz], './{}/train_multigrid_{}_{:02d}_{:06d}.jpg'.
                                    format(flags.sample_dir, to_sz, epoch, idx_batch))

                    print('Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f} , train_loss[multigrid_{}]: {:.5f}, reconstruction loss[multigrid_{}]: {:.5f}'.format(
                        epoch + 1, idx_batch + 1, batch_idxs, time.time() - start_time, to_sz, err, to_sz, err2))
                    from_sz = to_sz

                if np.mod(counter, 100) == 1:
                    tmp_feed_dict = {}
                    tmp_feed_dict[self.phase] = True
                    # save summary
                    for to_sz in self.scale_list[1:]:
                        tmp_feed_dict[self.original_images[to_sz]] = train_images[to_sz]
                        tmp_feed_dict[self.synthesized_images[to_sz]] = samples[to_sz]

                    summary = self.sess.run(self.mysummary, feed_dict=tmp_feed_dict)
                    self.writer.add_summary(summary, counter)

                counter += 1

                if np.mod(counter, 1000) == 2 or epoch == flags.epoch-1 and idx_batch == batch_idxs-1:
                    self.save(flags, counter)

            start_idx = 0

    def model_dir(self, flags):
        return '{}_{}'.format(flags.dataset_name, self.batch_size)

    def save(self, flags, step):
        model_name = 'multigrid.model'
        checkpoint_dir = os.path.join(flags.checkpoint_dir, self.model_dir(flags))

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, flags):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(flags.checkpoint_dir, self.model_dir(flags))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



