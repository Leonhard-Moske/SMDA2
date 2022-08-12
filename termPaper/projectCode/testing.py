# testing script with hacked together code

import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import os
import random
import astroML.datasets

tfd = tfp.distributions
tfb = tfp.bijectors

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# #load data into /tmp/astroML_data
# x_data, y_data = astroML.datasets.fetch_rrlyrae_combined(data_home="/tmp/astroML_data", download_if_missing=True)

# featurelabels = ["u-g","g-r", "r-i", "i-z"]

# xtfData = tf.data.Dataset.from_tensor_slices(x_data[y_data==1])
# print(next(iter(xtfData)))
# xtfData = xtfData.batch(batch_size=len(x_data)//10)

# xtfData2 = tf.data.Dataset.from_tensor_slices(x_data[y_data==0])
# print(next(iter(xtfData)))
# xtfData2 = xtfData2.batch(batch_size=len(x_data)//100)
# # print(next(iter(xtfData)))
# # print(len(x_data))

# hidden_shape = [200, 200]  # hidden shape for MADE network of MAF
# layers = 12  # number of layers of the flow

# base_dist = tfd.Normal(loc=0.0, scale=1.0)  # specify base distribution
# base_dist2 = tfd.Normal(loc=0.0, scale=1.0)  # specify base distribution


# tfk = tf.keras

# class Made(tfk.layers.Layer):
#     """
#     Implementation of a Masked Autoencoder for Distribution Estimation (MADE) [Germain et al. (2015)].
#     The existing TensorFlow bijector "AutoregressiveNetwork" is used. The output is reshaped to output one shift vector
#     and one log_scale vector.

#     :param params: Python integer specifying the number of parameters to output per input.
#     :param event_shape: Python list-like of positive integers (or a single int), specifying the shape of the input to this layer, which is also the event_shape of the distribution parameterized by this layer. Currently only rank-1 shapes are supported. That is, event_shape must be a single integer. If not specified, the event shape is inferred when this layer is first called or built.
#     :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
#     :param activation: An activation function. See tf.keras.layers.Dense. Default: None.
#     :param use_bias: Whether or not the dense layers constructed in this layer should have a bias term. See tf.keras.layers.Dense. Default: True.
#     :param kernel_regularizer: Regularizer function applied to the Dense kernel weight matrices. Default: None.
#     :param bias_regularizer: Regularizer function applied to the Dense bias weight vectors. Default: None.
#     """

#     def __init__(self, params, event_shape=None, hidden_units=None, activation=None, use_bias=True,
#                  kernel_regularizer=None, bias_regularizer=None, name="made"):

#         super(Made, self).__init__(name=name)

#         self.params = params
#         self.event_shape = event_shape
#         self.hidden_units = hidden_units
#         self.activation = activation
#         self.use_bias = use_bias
#         self.kernel_regularizer = kernel_regularizer
#         self.bias_regularizer = bias_regularizer

#         self.network = tfb.AutoregressiveNetwork(params=params, event_shape=event_shape, hidden_units=hidden_units,
#                                                  activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer, 
#                                                  bias_regularizer=bias_regularizer)

#     def call(self, x):
#         shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

#         return shift, tf.math.tanh(log_scale)


# bijectors = []
# for i in range(0, layers):
#     bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
#     bijectors.append(tfb.Permute(permutation=[3, 0, 1, 2]))  # data permutation after layers of MAF
    
# bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='chain_of_maf')

# bijectors2 = []
# for i in range(0, layers):
#     bijectors2.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
#     bijectors2.append(tfb.Permute(permutation=[3, 0, 1, 2]))  # data permutation after layers of MAF
    
# bijector2 = tfb.Chain(bijectors=list(reversed(bijectors2)), name='chain_of_maf')


# maf2 = tfd.TransformedDistribution(
#     distribution=tfd.Sample(base_dist2, sample_shape=[4]),
#     bijector=bijector2,
# )

# maf = tfd.TransformedDistribution(
#     distribution=tfd.Sample(base_dist, sample_shape=[4]),
#     bijector=bijector,
# )




# # initialize flow
# samples = maf.sample(10) #generate 10 samples

# print(maf.prob([0,1,0,1]))
# print(maf2.prob([0,1,0,1]))
# print(maf.distribution.prob([0,0,0,0]))
# # print(maf.bijector.inverse([0.0,1.0,0.0,1.0]))

# # print(samples)
# #print(maf.mean())

# @tf.function
# def train_density_estimation(distribution, optimizer, batch):
#     """
#     Train function for density estimation normalizing flows.
#     :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
#     :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
#     :param batch: Batch of the train data.
#     :return: loss.
#     """
#     with tf.GradientTape() as tape: #Gradient Tape is differentiation
#         tape.watch(distribution.trainable_variables) #define variables to differentiate with
#         loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
#         gradients = tape.gradient(loss, distribution.trainable_variables) # compute gradients of varaibles at loss 
#         optimizer.apply_gradients(zip(gradients, distribution.trainable_variables)) # apply the gradients

#         return loss

# @tf.function
# def train_density_estimation2(distribution, optimizer, batch):
#     """
#     Train function for density estimation normalizing flows.
#     :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
#     :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
#     :param batch: Batch of the train data.
#     :return: loss.
#     """
#     with tf.GradientTape() as tape: #Gradient Tape is differentiation
#         tape.watch(distribution.trainable_variables) #define variables to differentiate with
#         loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
#         gradients = tape.gradient(loss, distribution.trainable_variables) # compute gradients of varaibles at loss 
#         optimizer.apply_gradients(zip(gradients, distribution.trainable_variables)) # apply the gradients

#         return loss

# @tf.function
# def nll(distribution, data):
#     """
#     Computes the negative log liklihood loss for a given distribution and given data.
#     :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
#     :param data: Data or a batch from data.
#     :return: Negative Log Likelihodd loss.
#     """
#     return -tf.reduce_mean(distribution.log_prob(data))

# def plot_heatmap_2d(dist, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=1000, name=None):
#     plt.figure()
    
#     x = tf.linspace(xmin, xmax, mesh_count)
#     y = tf.linspace(ymin, ymax, mesh_count)
#     X, Y = tf.meshgrid(x, y)
    
#     concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1]), [0.0]*40000, [0.0]*40000])) # 0,0 for 4 dim 
#     prob = dist.prob(concatenated_mesh_coordinates)
#     #plt.hexbin(concatenated_mesh_coordinates[:,0], concatenated_mesh_coordinates[:,1], C=prob, cmap='rainbow')
#     prob = prob.numpy()
    
#     plt.imshow(tf.transpose(tf.reshape(prob, (mesh_count, mesh_count))), origin="lower")
#     plt.xticks([0, mesh_count * 0.25, mesh_count * 0.5, mesh_count * 0.75, mesh_count], [xmin, xmin/2, 0, xmax/2, xmax])
#     plt.yticks([0, mesh_count * 0.25, mesh_count * 0.5, mesh_count * 0.75, mesh_count], [ymin, ymin/2, 0, ymax/2, ymax])
#     if name:
#         plt.savefig(name + ".png", format="png")


# base_lr = 1e-3
# end_lr = 1e-4
# max_epochs = int(100)  # maximum number of epochs of the training
# learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)

# # initialize checkpoints
# checkpoint_directory = "{}/tmp_{}".format("coolData", str(hex(random.getrandbits(32))))
# checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

# opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
# opt2 = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
# checkpoint = tf.train.Checkpoint(optimizer=opt, model=maf)

# global_step = []
# train_losses = []
# val_losses = []
# min_val_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)  # high value to ensure that first loss < min_loss
# min_train_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)
# min_val_epoch = 0
# min_train_epoch = 0
# delta_stop = 1000  # threshold for early stopping

# t_start = time.time()  # start time

# # start training
# for i in range(max_epochs):
#     print(i)
#     for batch in xtfData:
#         train_loss = train_density_estimation(maf, opt, batch)
#     print(train_loss)

#     if i % int(100) == 0:
#         #val_loss = nll(maf, val_data)
#         global_step.append(i)
#         train_losses.append(train_loss)
#         #val_losses.append(val_loss)
#         #print(f"{i}, train_loss: {train_loss}, val_loss: {val_loss}")

#         if train_loss < min_train_loss:
#             min_train_loss = train_loss
#             min_train_epoch = i

#         # if val_loss < min_val_loss:
#         #     min_val_loss = val_loss
#         #     min_val_epoch = i
#         #     checkpoint.write(file_prefix=checkpoint_prefix)  # overwrite best val model

#         elif i - min_val_epoch > delta_stop:  # no decrease in min_val_loss for "delta_stop epochs"
#             break

#     #if i % int(10) == 0:
#         ## plot heatmap every 1000 epochs
#         #plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, name="testingpng")


# max_epochs = int(6)

# for i in range(max_epochs):
#     print(i)
#     for batch in xtfData2:
#         train_loss = train_density_estimation2(maf2, opt2, batch)
#     print(train_loss)

#     if i % int(100) == 0:
#         #val_loss = nll(maf, val_data)
#         global_step.append(i)
#         train_losses.append(train_loss)
#         #val_losses.append(val_loss)
#         #print(f"{i}, train_loss: {train_loss}, val_loss: {val_loss}")

#         if train_loss < min_train_loss:
#             min_train_loss = train_loss
#             min_train_epoch = i

#         # if val_loss < min_val_loss:
#         #     min_val_loss = val_loss
#         #     min_val_epoch = i
#         #     checkpoint.write(file_prefix=checkpoint_prefix)  # overwrite best val model

#         elif i - min_val_epoch > delta_stop:  # no decrease in min_val_loss for "delta_stop epochs"
#             break

#     #if i % int(10) == 0:
#         ## plot heatmap every 1000 epochs
#         #plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, name="testingpng")


# train_time = time.time() - t_start

# # evaluate the distributions

# def classify(dist1, dist2, data): #returns the test statistic ln(p1(data)/p2(data))
#     #print(data)
#     prob1 = dist1.log_prob(data)
#     prob2 = dist2.log_prob(data)
#     #print(dist1.prob(data))
#     #print(dist2.prob(data))
#     return prob1 - prob2

# xtfData = xtfData.batch(batch_size=1)
# xtfData2 = xtfData2.batch(batch_size=1)



# for i in range(1):
#     print(classify(maf, maf2, next(iter(xtfData))))

# for i in range(1):
#     print(classify(maf, maf2, next(iter(xtfData2))))



class testbij(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name="testbij") -> None:
        super(testbij, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims = 0,
            name = name
        )
    def _forward(self,x):
        return tf.exp(x)

    def _inverse(self,y):
        return tf.log(y)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
      # Notice that we needn't do any reducing, even when`event_ndims > 0`.
      # The base Bijector class will handle reducing for us; it knows how
      # to do so because we called `super` `__init__` with
      # `forward_min_event_ndims = 0`.
      return x

a = testbij()

print(a._forward([0.,1.,2.]))


import keras.layers as kl
import keras

class NN(keras.Layer):
    """
    Neural Network Architecture for calcualting s and t for Real-NVP
    
    :param input_shape: shape of the data coming in the layer
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: Activation of the hidden units
    """
    def __init__(self, input_shape, n_hidden=[512, 512], activation="relu", name="nn"):
        super(NN, self).__init__(name="nn")
        layer_list = []
        for i, hidden in enumerate(n_hidden):
            layer_list.append(kl.Dense(hidden, activation=activation))
        self.layer_list = layer_list
        self.log_s_layer = kl.Dense(input_shape, activation="tanh", name='log_s')
        self.t_layer = kl.Dense(input_shape, name='t')

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        t = self.t_layer(y)
        return log_s, t

from keras import Model 

class RealNVP(tfb.Bijector):
    """
    Implementation of a Real-NVP for Denisty Estimation. L. Dinh “Density estimation using Real NVP,” 2016.
    This implementation only works for 1D arrays.
    :param input_shape: shape of the data coming in the layer
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.

    """

    def __init__(self, input_shape, n_hidden=[512, 512], forward_min_event_ndims=1, validate_args: bool = False, name="real_nvp"):
        super(RealNVP, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )

        assert input_shape % 2 == 0
        input_shape = input_shape // 2
        nn_layer = NN(input_shape, n_hidden)
        x = tf.keras.Input(input_shape)
        log_s, t = nn_layer(x)
        self.nn = Model(x, [log_s, t], name="nn")
        
    def _bijector_fn(self, x):
        log_s, t = self.nn(x)
        return tfb.affine_scalar.AffineScalar(shift=t, log_scale=log_s)

    def _forward(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        y_a = self._bijector_fn(x_b).forward(x_a)
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        return self._bijector_fn(x_b).forward_log_det_jacobian(x_a, event_ndims=1)
    
    def _inverse_log_det_jacobian(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        return self._bijector_fn(y_b).inverse_log_det_jacobian(y_a, event_ndims=1)