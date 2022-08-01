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

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

#load data into /tmp/astroML_data
x_data, y_data = astroML.datasets.fetch_rrlyrae_combined(data_home="/tmp/astroML_data", download_if_missing=True)

featurelabels = ["u-g","g-r", "r-i", "i-z"]

x_data1 = tf.convert_to_tensor(x_data[y_data == 1], dtype=np.float32)

print(x_data)

hidden_shape = [200, 200]  # hidden shape for MADE network of MAF
layers = 12  # number of layers of the flow

base_dist = tfd.Normal(loc=0.0, scale=1.0)  # specify base distribution


tfk = tf.keras

class Made(tfk.layers.Layer):
    """
    Implementation of a Masked Autoencoder for Distribution Estimation (MADE) [Germain et al. (2015)].
    The existing TensorFlow bijector "AutoregressiveNetwork" is used. The output is reshaped to output one shift vector
    and one log_scale vector.

    :param params: Python integer specifying the number of parameters to output per input.
    :param event_shape: Python list-like of positive integers (or a single int), specifying the shape of the input to this layer, which is also the event_shape of the distribution parameterized by this layer. Currently only rank-1 shapes are supported. That is, event_shape must be a single integer. If not specified, the event shape is inferred when this layer is first called or built.
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: An activation function. See tf.keras.layers.Dense. Default: None.
    :param use_bias: Whether or not the dense layers constructed in this layer should have a bias term. See tf.keras.layers.Dense. Default: True.
    :param kernel_regularizer: Regularizer function applied to the Dense kernel weight matrices. Default: None.
    :param bias_regularizer: Regularizer function applied to the Dense bias weight vectors. Default: None.
    """

    def __init__(self, params, event_shape=None, hidden_units=None, activation=None, use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, name="made"):

        super(Made, self).__init__(name=name)

        self.params = params
        self.event_shape = event_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.network = tfb.AutoregressiveNetwork(params=params, event_shape=event_shape, hidden_units=hidden_units,
                                                 activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer, 
                                                 bias_regularizer=bias_regularizer)

    def call(self, x):
        shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)


bijectors = []
for i in range(0, layers):
    bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
    bijectors.append(tfb.Permute(permutation=[1, 0, 2, 3]))  # data permutation after layers of MAF
    
bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='chain_of_maf')

maf = tfd.TransformedDistribution(
    distribution=tfd.Sample(base_dist, sample_shape=[4]),
    bijector=bijector,
)

# initialize flow
samples = maf.sample(10) #generate 10 samples

print(maf.prob([0,1,0,1]))
print(maf.bijector.inverse([0.0,1.0,0.0,1.0]))

print(samples)
#print(maf.mean())

@tf.function
def train_density_estimation(distribution, optimizer, batch):
    """
    Train function for density estimation normalizing flows.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    with tf.GradientTape() as tape: #Gradient Tape is differentiation
        tape.watch(distribution.trainable_variables) #define variables to differentiate with
        loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
    gradients = tape.gradient(loss, distribution.trainable_variables) # compute gradients of varaibles at loss 
    optimizer.apply_gradients(zip(gradients, distribution.trainable_variables)) # apply the gradients

    return loss

@tf.function
def nll(distribution, data):
    """
    Computes the negative log liklihood loss for a given distribution and given data.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param data: Data or a batch from data.
    :return: Negative Log Likelihodd loss.
    """
    return -tf.reduce_mean(distribution.log_prob(data))

def plot_heatmap_2d(dist, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=1000, name=None):
    plt.figure()
    
    x = tf.linspace(xmin, xmax, mesh_count)
    y = tf.linspace(ymin, ymax, mesh_count)
    X, Y = tf.meshgrid(x, y)
    
    concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1]), [0.0]*40000, [0.0]*40000])) # 0,0 for 4 dim 
    prob = dist.prob(concatenated_mesh_coordinates)
    #plt.hexbin(concatenated_mesh_coordinates[:,0], concatenated_mesh_coordinates[:,1], C=prob, cmap='rainbow')
    prob = prob.numpy()
    
    plt.imshow(tf.transpose(tf.reshape(prob, (mesh_count, mesh_count))), origin="lower")
    plt.xticks([0, mesh_count * 0.25, mesh_count * 0.5, mesh_count * 0.75, mesh_count], [xmin, xmin/2, 0, xmax/2, xmax])
    plt.yticks([0, mesh_count * 0.25, mesh_count * 0.5, mesh_count * 0.75, mesh_count], [ymin, ymin/2, 0, ymax/2, ymax])
    if name:
        plt.savefig(name + ".png", format="png")


base_lr = 1e-3
end_lr = 1e-4
max_epochs = int(5e3)  # maximum number of epochs of the training
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)

# initialize checkpoints
checkpoint_directory = "{}/tmp_{}".format("coolData", str(hex(random.getrandbits(32))))
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
checkpoint = tf.train.Checkpoint(optimizer=opt, model=maf)

global_step = []
train_losses = []
val_losses = []
min_val_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)  # high value to ensure that first loss < min_loss
min_train_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)
min_val_epoch = 0
min_train_epoch = 0
delta_stop = 1000  # threshold for early stopping

t_start = time.time()  # start time

# start training
for i in range(max_epochs):
    print(i)
    for batch in x_data1:
        train_loss = train_density_estimation(maf, opt, batch)
    print(train_loss)

    if i % int(100) == 0:
        #val_loss = nll(maf, val_data)
        global_step.append(i)
        train_losses.append(train_loss)
        #val_losses.append(val_loss)
        #print(f"{i}, train_loss: {train_loss}, val_loss: {val_loss}")

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_epoch = i

        # if val_loss < min_val_loss:
        #     min_val_loss = val_loss
        #     min_val_epoch = i
        #     checkpoint.write(file_prefix=checkpoint_prefix)  # overwrite best val model

        elif i - min_val_epoch > delta_stop:  # no decrease in min_val_loss for "delta_stop epochs"
            break

    if i % int(10) == 0:
        # plot heatmap every 1000 epochs
        plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, name="testingpng")

train_time = time.time() - t_start