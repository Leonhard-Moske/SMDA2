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
tfk = tf.keras

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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


@tf.function
def train_density_estimation0(distribution, optimizer, batch):
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
def train_density_estimation1(distribution, optimizer, batch):
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

def shuffle_split(samples, train_split, val_split):
    '''
    Shuffles the data and performs a train-validation-test split.
    Test = 1 - (train + val).
    
    :param samples: Samples from a dataset / data distribution.
    :param train: Portion of the samples used for training (float32, 0<=train<1).
    :param val: Portion of the samples used for validation (float32, 0<=val<1).
    :return train_data, val_data, test_data: 
    '''

    if train_split + val_split > 1:
        raise Exception('train_split plus val_split has to be smaller or equal to one.')

    batch_size = len(samples)
    np.random.shuffle(samples)
    n_train = int(round(train_split * batch_size))
    n_val = int(round((train_split + val_split) * batch_size))
    train_data = tf.cast(samples[0:n_train], dtype=tf.float32)
    val_data = tf.cast(samples[n_train:n_val], dtype=tf.float32)
    test_data = tf.cast(samples[n_val:batch_size], dtype=tf.float32)

    return train_data, val_data, test_data

#------------------------------------------------------------------------------


train_split = 0.8
val_split = 0.1
batchsizes = [1000, 100]

hidden_shape= [200, 200]
layers = 12

base_lr = 1e-3
end_lr = 1e-4
max_epochs = [int(10), int(100)]  # maximum number of epochs of the training

#------------------------------------------------------------------------------


#load data into /tmp/astroML_data
x_data, y_data = astroML.datasets.fetch_rrlyrae_combined(data_home="/tmp/astroML_data", download_if_missing=True)

featurelabels = ["u-g","g-r", "r-i", "i-z"]

train_data0, batched_val_data0, batched_test_data0 = shuffle_split(x_data[y_data == 0], train_split, val_split)
train_data1, batched_val_data1, batched_test_data1 = shuffle_split(x_data[y_data == 1], train_split, val_split)

train_data_batched0 = tf.data.Dataset.from_tensor_slices(train_data0).batch(batch_size=batchsizes[0])
train_data_batched1 = tf.data.Dataset.from_tensor_slices(train_data1).batch(batch_size=batchsizes[1])

#------------------------------------------------------------------------------

base_dist0 = tfd.Normal(loc=0.0, scale=1.0)  # specify base distribution
base_dist1 = tfd.Normal(loc=0.0, scale=1.0)  # specify base distribution



bijectors0 = []
for i in range(0, layers):
    bijectors0.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
    bijectors0.append(tfb.Permute(permutation=[3, 0, 1, 2]))  # data permutation after layers of MAF
    
bijector0 = tfb.Chain(bijectors=list(reversed(bijectors0)), name='bijector0')

bijectors1 = []
for i in range(0, layers):
    bijectors1.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
    bijectors1.append(tfb.Permute(permutation=[3, 0, 1, 2]))  # data permutation after layers of MAF
    
bijector1 = tfb.Chain(bijectors=list(reversed(bijectors1)), name='bijector1')

flow0 = tfd.TransformedDistribution(
    distribution=tfd.Sample(base_dist0, sample_shape=[4]),
    bijector=bijector0,
)

flow1 = tfd.TransformedDistribution(
    distribution=tfd.Sample(base_dist1, sample_shape=[4]),
    bijector=bijector1,
)

#------------------------------------------------------------------------------


learning_rate_fn0 = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs[0], end_lr, power=0.5)
learning_rate_fn1 = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs[1], end_lr, power=0.5)


opt0 = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn0)  # optimizer
opt1 = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn1)  # optimizer


global_step = []
train_losses = []
val_losses = []
min_val_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)  # high value to ensure that first loss < min_loss
min_train_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)
min_val_epoch = 0
min_train_epoch = 0
delta_stop = 1000  # threshold for early stopping

for i in range(max_epochs[0]):
    print(i)
    for batch in train_data_batched0:
        train_loss = train_density_estimation0(flow0, opt0, batch)
    print(train_loss)

for i in range(max_epochs[1]):
    print(i)
    for batch in train_data_batched1:
        train_loss = train_density_estimation1(flow1, opt1, batch)
    print(train_loss)

#------------------------------------------------------------------------------

def classify(dist1, dist2, data): #returns the test statistic ln(p1(data)/p2(data))
    #print(data)
    prob1 = dist1.log_prob(data)
    prob2 = dist2.log_prob(data)
    #print(dist1.prob(data))
    #print(dist2.prob(data))
    return prob1 - prob2

validation0=classify(flow0, flow1, batched_val_data0)
validation1=classify(flow0, flow1, batched_val_data1)

print(validation0)
print(validation1)
