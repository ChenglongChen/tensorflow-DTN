
import os
import numpy as np
from scipy.misc import imsave

import tensorflow as tf
from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
    """
    Code: https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
    """
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


class DomainTransferNet(object):

    def __init__(self, params, logger):
        self.params = params
        self.logger = logger
        self.init_tf_vars()
        self.build_f()
        self.flip_gradient = True if ("flip_gradient" in self.params and self.params["flip_gradient"]) else False
        if self.flip_gradient:
            self.build_main_model_with_flip_gradient()
        else:
            self.build_main_model()
        self.sess, self.saver = self._init_session()
        self.summary_writer = tf.summary.FileWriter(self.params["summary_dir"], self.sess.graph)


    def init_tf_vars(self):
        self.xs          = tf.placeholder(tf.float32, [None, 32, 32, 3], "xs")
        self.xt          = tf.placeholder(tf.float32, [None, 32, 32, 1], "xt")
        self.labels      = tf.placeholder(tf.int64, [None], "labels")
        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        self.len_xs      = tf.shape(self.xs)[0]
        self.len_xt      = tf.shape(self.xt)[0]


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 1})
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        # max_to_keep=None, keep all the models
        saver = tf.train.Saver(max_to_keep=None)
        return sess, saver


    def save_session(self):
        self.saver.save(self.sess, self.params["offline_model_dir"] + "/self.checkpoint")


    def restore_session(self):
        self.saver.restore(self.sess, self.params["offline_model_dir"] + "/self.checkpoint")


    def conv_bn(self, x, num_filters, kernel_size, stride, padding, activation, bn, name, is_training):
        x = tf.layers.conv2d(
            inputs=x,
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=None,
            strides=stride,
            reuse=tf.AUTO_REUSE,
            name=name)
        if bn:
            x = tf.layers.BatchNormalization()(x, is_training)
        x = activation(x)
        return x


    def conv_t_bn(self, x, num_filters, kernel_size, stride, padding, activation, bn, name, is_training):
        x = tf.layers.conv2d_transpose(
            inputs=x,
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=None,
            strides=stride,
            reuse=tf.AUTO_REUSE,
            name=name)
        if bn:
            x = tf.layers.BatchNormalization()(x, is_training)
        x = activation(x)
        return x


    def f(self, x, bn=False):
        with tf.variable_scope("f", reuse=tf.AUTO_REUSE):
            x = tf.image.grayscale_to_rgb(x) if x.get_shape()[3] == 1 else x                        # (batch_size, 32, 32,   3)
            x = self.conv_bn(x,  64, [3, 3], 2,  "same", tf.nn.relu, bn, "conv1", self.is_training) # (batch_size, 16, 16,  64)
            x = self.conv_bn(x, 128, [3, 3], 2,  "same", tf.nn.relu, bn, "conv2", self.is_training) # (batch_size,  8,  8, 128)
            x = self.conv_bn(x, 256, [3, 3], 2,  "same", tf.nn.relu, bn, "conv3", self.is_training) # (batch_size,  4,  4, 256)
            x = self.conv_bn(x, 128, [4, 4], 2, "valid", tf.nn.tanh, bn, "conv4", self.is_training) # (batch_size,  1,  1, 128)
            return x

                
    def g(self, x, bn=False):
        with tf.variable_scope("g", reuse=tf.AUTO_REUSE):
            x = self.conv_t_bn(x, 512, [4, 4], 2, "valid", tf.nn.relu, bn,    "conv_t1", self.is_training) # (batch_size,  4,  4, 512)
            x = self.conv_t_bn(x, 256, [3, 3], 2,  "same", tf.nn.relu, bn,    "conv_t2", self.is_training) # (batch_size,  8,  8, 256)
            x = self.conv_t_bn(x, 128, [3, 3], 2,  "same", tf.nn.relu, bn,    "conv_t3", self.is_training) # (batch_size, 16, 16, 128)
            x = self.conv_t_bn(x,   1, [3, 3], 2,  "same", tf.nn.tanh, False, "conv_t4", self.is_training) # (batch_size, 32, 32,   1)
            return x
        
        
    def G(self, x):
        return self.g(self.f(x))

    
    def D(self, x, bn=False):
        with tf.variable_scope("D", reuse=tf.AUTO_REUSE):
            x = self.conv_bn(x, 128, [3, 3], 2,  "same", tf.nn.relu, bn,    "conv1", self.is_training) # (batch_size, 16, 16, 128)
            x = self.conv_bn(x, 256, [3, 3], 2,  "same", tf.nn.relu, bn,    "conv2", self.is_training) # (batch_size,  8,  8, 256)
            x = self.conv_bn(x, 512, [3, 3], 2,  "same", tf.nn.relu, bn,    "conv3", self.is_training) # (batch_size,  4,  4, 512)
            x = self.conv_bn(x,   1, [4, 4], 2, "valid", tf.nn.relu, False, "conv4", self.is_training) # (batch_size,  1,  1,   3)
            x = tf.layers.flatten(x)
            return x


    def build_f(self):

        f_xs = self.f(self.xs)
        logits = tf.layers.flatten(self.conv_bn(f_xs, 10, [1, 1], 1, "valid", tf.identity, False, "logits", False))

        # loss
        self.loss_auxiliary = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels))

        # accuracy
        preds = tf.argmax(logits, 1)
        self.acc_auxiliary = tf.reduce_mean(tf.cast(tf.equal(preds, self.labels), tf.float32))

        # train op
        optimizer = tf.train.AdamOptimizer(self.params["learning_rate"])
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        self.train_op_auxiliary = optimizer.minimize(self.loss_auxiliary)

        # summary op
        summary_loss = tf.summary.scalar("loss_auxiliary", self.loss_auxiliary)
        summary_acc = tf.summary.scalar("acc_auxiliary", self.acc_auxiliary)
        self.summary_op_auxiliary = tf.summary.merge([summary_loss, summary_acc])


    def build_main_model(self):

        # source domain
        f_xs        = self.f(self.xs)
        self.g_f_xs = self.g(f_xs)
        D_g_f_xs    = self.D(self.g_f_xs)
        f_g_f_xs    = self.f(self.g_f_xs)

        # target domain
        f_xt     = self.f(self.xt)
        g_f_xt   = self.g(f_xt)
        D_g_f_xt = self.D(g_f_xt)
        D_xt     = self.D(self.xt)

        # discriminator loss
        loss_D_g_f_xs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_g_f_xs, labels=tf.zeros_like(D_g_f_xs)))
        loss_D_g_f_xt = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_g_f_xt, labels=tf.zeros_like(D_g_f_xt)))
        loss_D_xt     = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_xt, labels=tf.ones_like(D_xt)))
        self.loss_D_xs   = loss_D_g_f_xs
        self.loss_D_xt   = loss_D_g_f_xt + loss_D_xt

        # generator loss
        loss_GANG_D_g_f_xs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_g_f_xs, labels=tf.ones_like(D_g_f_xs)))
        loss_GANG_D_g_f_xt = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_g_f_xt, labels=tf.ones_like(D_g_f_xt)))
        # loss_GANG          = loss_GANG_D_g_f_xs + loss_GANG_D_g_f_xt
        loss_CONST         = tf.reduce_mean(tf.square(f_xs - f_g_f_xs))
        loss_TID           = tf.reduce_mean(tf.square(self.xt - g_f_xt))
        self.loss_G_xs     = loss_GANG_D_g_f_xs + self.params["loss_const_weight"] * loss_CONST
        self.loss_G_xt     = loss_GANG_D_g_f_xt + self.params["loss_tid_weight"] * loss_TID

        # train op
        t_vars = tf.trainable_variables()
        d_vars = [v for v in t_vars if "D" in v.name]
        g_vars = [v for v in t_vars if "g" in v.name]

        with tf.variable_scope("xs", reuse=False):
            optimizer_d_xs = tf.train.AdamOptimizer(self.params["learning_rate"])
            optimizer_g_xs = tf.train.AdamOptimizer(self.params["learning_rate"])
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            self.train_op_d_xs = optimizer_d_xs.minimize(self.loss_D_xs, var_list=d_vars)
            self.train_op_g_xs = optimizer_g_xs.minimize(self.loss_G_xs, var_list=g_vars)

        with tf.variable_scope("xt", reuse=False):
            optimizer_d_xt = tf.train.AdamOptimizer(self.params["learning_rate"])
            optimizer_g_xt = tf.train.AdamOptimizer(self.params["learning_rate"])
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            self.train_op_d_xt = optimizer_d_xt.minimize(self.loss_D_xt, var_list=d_vars)
            self.train_op_g_xt = optimizer_g_xt.minimize(self.loss_G_xt, var_list=g_vars)

        # summary op
        summary_loss_D_xs = tf.summary.scalar("loss_D_xs", self.loss_D_xs)
        summary_loss_G_xs = tf.summary.scalar("loss_G_xs", self.loss_G_xs)
        summary_xs        = tf.summary.image("xs", self.xs)
        summary_g_f_xs    = tf.summary.image("g_f_xs", self.g_f_xs)

        summary_loss_D_xt = tf.summary.scalar("loss_D_xt", self.loss_D_xt)
        summary_loss_G_xt = tf.summary.scalar("loss_G_xt", self.loss_G_xt)
        summary_xt        = tf.summary.image("xt", self.xt)
        summary_g_f_xt    = tf.summary.image("g_f_xt", g_f_xt)

        self.summary_op_xs = tf.summary.merge([summary_loss_D_xs, summary_loss_G_xs, summary_xs, summary_g_f_xs])
        self.summary_op_xt = tf.summary.merge([summary_loss_D_xt, summary_loss_G_xt, summary_xt, summary_g_f_xt])


    def build_main_model_with_flip_gradient(self):

        self.flip_gradient_ = FlipGradientBuilder()

        # source domain
        f_xs        = self.f(self.xs)
        self.g_f_xs = self.g(f_xs)
        D_g_f_xs    = self.D(self.flip_gradient_(self.g_f_xs))
        f_g_f_xs    = self.f(self.g_f_xs)

        # target domain
        f_xt     = self.f(self.xt)
        g_f_xt   = self.g(f_xt)
        D_g_f_xt = self.D(self.flip_gradient_(g_f_xt))
        D_xt     = self.D(self.xt)

        # discriminator loss
        loss_D_g_f_xs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_g_f_xs, labels=tf.zeros_like(D_g_f_xs)))
        loss_D_g_f_xt = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_g_f_xt, labels=tf.zeros_like(D_g_f_xt)))
        loss_D_xt     = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_xt, labels=tf.ones_like(D_xt)))
        self.loss_D_xs   = loss_D_g_f_xs
        self.loss_D_xt   = loss_D_g_f_xt + loss_D_xt
        self.loss_D      = self.loss_D_xs + self.loss_D_xt

        # generator loss
        loss_GANG_D_g_f_xs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_g_f_xs, labels=tf.ones_like(D_g_f_xs)))
        loss_GANG_D_g_f_xt = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_g_f_xt, labels=tf.ones_like(D_g_f_xt)))
        loss_GANG          = loss_GANG_D_g_f_xs + loss_GANG_D_g_f_xt
        loss_GANG_D_g_f_xs = 0.
        loss_GANG_D_g_f_xt = 0.
        loss_GANG          = 0.
        loss_CONST         = tf.reduce_mean(tf.square(f_xs - f_g_f_xs))
        loss_TID           = tf.reduce_mean(tf.square(self.xt - g_f_xt))
        self.loss_G_xs     = loss_GANG_D_g_f_xs + self.params["loss_const_weight"] * loss_CONST
        self.loss_G_xt     = loss_GANG_D_g_f_xt + self.params["loss_tid_weight"] * loss_TID
        self.loss_G        = self.loss_G_xs + self.loss_G_xt

        self.loss_G_D      = self.loss_G + self.loss_D

        # train op
        t_vars = tf.trainable_variables()
        d_g_vars = [v for v in t_vars if ("D" in v.name) or ("g" in v.name)]

        with tf.variable_scope("all", reuse=False):
            optimizer = tf.train.AdamOptimizer(self.params["learning_rate"])
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            self.train_op_all = optimizer.minimize(self.loss_G_D, var_list=d_g_vars)

        # summary op
        summary_loss_D_xs = tf.summary.scalar("loss_D_xs", self.loss_D_xs)
        summary_loss_G_xs = tf.summary.scalar("loss_G_xs", self.loss_G_xs)
        summary_xs        = tf.summary.image("xs", self.xs)
        summary_g_f_xs    = tf.summary.image("g_f_xs", self.g_f_xs)

        summary_loss_D_xt = tf.summary.scalar("loss_D_xt", self.loss_D_xt)
        summary_loss_G_xt = tf.summary.scalar("loss_G_xt", self.loss_G_xt)
        summary_xt        = tf.summary.image("xt", self.xt)
        summary_g_f_xt    = tf.summary.image("g_f_xt", g_f_xt)

        self.summary_op_xs = tf.summary.merge([summary_loss_D_xs, summary_loss_G_xs, summary_xs, summary_g_f_xs])
        self.summary_op_xt = tf.summary.merge([summary_loss_D_xt, summary_loss_G_xt, summary_xt, summary_g_f_xt])


    def fit(self, auxiliary_data, Xs_train, Xt_train):
        
        #### pretrain auxiliary classifier for f
        for batch in range(self.params["max_batch_pretrain"] + 1):
            i = batch % int(auxiliary_data["X_train"].shape[0] / self.params["batch_size"])
            xs = auxiliary_data["X_train"][i * self.params["batch_size"]:(i + 1) * self.params["batch_size"]]
            labels = auxiliary_data["y_train"][i * self.params["batch_size"]:(i + 1) * self.params["batch_size"]]
            feed_dict = {
                self.xs: xs,
                self.labels: labels,
                self.is_training: True,
            }
            self.sess.run(self.train_op_auxiliary, feed_dict)

            if (batch + 1) % self.params["eval_every_num_update_pretrain"] == 0:
                summary, loss_auxiliary, acc_auxiliary = self.sess.run(
                    [self.summary_op_auxiliary, self.loss_auxiliary, self.acc_auxiliary], feed_dict)
                rand_idxs = np.random.permutation(auxiliary_data["X_test"].shape[0])[:self.params["batch_size"]]
                feed_dict = {
                    self.xs: auxiliary_data["X_test"][rand_idxs],
                    self.labels: auxiliary_data["y_test"][rand_idxs],
                    self.is_training: False,
                }
                acc_auxiliary_test = self.sess.run(self.acc_auxiliary, feed_dict)
                self.summary_writer.add_summary(summary, batch)
                self.logger.info("[%d/%d] train-loss: %.6f, train-acc: %.6f, test-acc: %.6f" \
                      % (batch + 1, self.params["max_batch_pretrain"], loss_auxiliary, acc_auxiliary, acc_auxiliary_test))


        #### train domain transfer network
        for batch in range(self.params["max_batch"] + 1):

            i = batch % int(Xs_train.shape[0] / self.params["batch_size"])
            xs = Xs_train[i * self.params["batch_size"]:(i + 1) * self.params["batch_size"]]
            j = batch % int(Xt_train.shape[0] / self.params["batch_size"])
            xt = Xt_train[j * self.params["batch_size"]:(j + 1) * self.params["batch_size"]]
            feed_dict = {
                self.xs: xs,
                self.xt: xt,
                self.is_training: True,
            }

            if self.flip_gradient:
                self.sess.run(self.train_op_all, feed_dict)
            else:
                self.sess.run(self.train_op_d_xs, feed_dict)
                self.sess.run(self.train_op_g_xs, feed_dict)
                self.sess.run(self.train_op_g_xs, feed_dict)
                self.sess.run(self.train_op_g_xs, feed_dict)
                self.sess.run(self.train_op_g_xs, feed_dict)
                self.sess.run(self.train_op_g_xs, feed_dict)
                self.sess.run(self.train_op_g_xs, feed_dict)

            if (batch + 1) % self.params["eval_every_num_update"] == 0:
                op = [self.summary_op_xs, self.loss_D_xs, self.loss_G_xs]
                summary_op_xs, loss_D_xs, loss_G_xs = self.sess.run(op, feed_dict)
                self.summary_writer.add_summary(summary_op_xs, batch)
                self.logger.info("[Source] [%d/%d] d_loss: %.6f, g_loss: %.6f" \
                      % (batch + 1, self.params["max_batch"], loss_D_xs, loss_G_xs))

            if self.flip_gradient:
                self.sess.run(self.train_op_all, feed_dict)
            else:
                self.sess.run(self.train_op_d_xt, feed_dict)
                self.sess.run(self.train_op_d_xt, feed_dict)
                self.sess.run(self.train_op_g_xt, feed_dict)
                self.sess.run(self.train_op_g_xt, feed_dict)
                self.sess.run(self.train_op_g_xt, feed_dict)
                self.sess.run(self.train_op_g_xt, feed_dict)

            if (batch + 1) % self.params["eval_every_num_update"] == 0:
                op = [self.summary_op_xt, self.loss_D_xt, self.loss_G_xt]
                summary_op_xt, loss_D_xt, loss_G_xt = self.sess.run(op, feed_dict)
                self.summary_writer.add_summary(summary_op_xt, batch)
                self.logger.info("[Target] [%d/%d] d_loss: %.6f, g_loss: %.6f" \
                      % (batch + 1, self.params["max_batch"], loss_D_xt, loss_G_xt))


    def merge_images(self, sources, targets, batch_size):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(batch_size))
        merged = np.zeros([row*h, row*w*2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i*h:(i+1)*h, (j*2)*h:(j*2+1)*h, :] = s
            merged[i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h, :] = t
        return merged


    def evaluate(self, Xs, sample_batch, batch_size, sample_dir):

        self.logger.info("start evaluate")
        for i in range(sample_batch):
            xs = Xs[i*batch_size:(i+1)*batch_size]
            feed_dict = {
                self.xs: xs,
                self.is_training: False,
            }
            g_f_xs = self.sess.run(self.g_f_xs, feed_dict)

            # merge and save source images and sampled target images
            merged = self.merge_images(xs, g_f_xs, batch_size)
            path = os.path.join(sample_dir, "sample-%d-to-%d.png" %(i*batch_size, (i+1)*batch_size))
            imsave(path, merged)
            self.logger.info("saved %s" %path)