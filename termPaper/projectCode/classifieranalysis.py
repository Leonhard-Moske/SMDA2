import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import os
import random
import astroML.datasets
from sklearn.metrics import roc_curve
import keras.layers as kl
import keras
from keras import Model 


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



class NN(tfk.layers.Layer):
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
        return tfb.Chain([tfb.Scale(log_scale=log_s),tfb.Shift(shift = t)])


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

#------------------------------------------------------------------------------


train_split = 0.8
val_split = 0.1
batchsizes = [1000, 100]

hidden_shape= [200, 200]
layers = 8
n_hidden = [100, 100, 100, 100]

base_lr = 1e-3
end_lr = 1e-4
max_epochs = [int(20), int(200)]  # maximum number of epochs of the training

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
    bijectors0.append(RealNVP(input_shape= (4), n_hidden= n_hidden))
    #bijectors0.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
    bijectors0.append(tfb.Permute(permutation=[3, 0, 1, 2]))  # data permutation after layers of MAF
    
bijector0 = tfb.Chain(bijectors=list(reversed(bijectors0)), name='bijector0')

bijectors1 = []
for i in range(0, layers):
    bijectors1.append(RealNVP(input_shape= (4), n_hidden= n_hidden))
    #bijectors1.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
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


train_losses = [[],[]]
val_losses = []


for i in range(max_epochs[0]):
    print(i)
    for batch in train_data_batched0:
        train_loss = train_density_estimation0(flow0, opt0, batch)
    train_losses[0].append(train_loss)
    print(train_loss)

for i in range(max_epochs[1]):
    print(i)
    for batch in train_data_batched1:
        train_loss = train_density_estimation1(flow1, opt1, batch)
    train_losses[1].append(train_loss)
    print(train_loss)

#------------------------------------------------------------------------------

def classify(dist1, dist2, data): #returns the test statistic ln(p1(data)/p2(data))
    prob1 = dist1.log_prob(data)
    prob2 = dist2.log_prob(data)
    return prob1 - prob2

def proportionRight(data, signal, flow0, flow1, cut= 0): #returns proportion of correct classified data
    response = classify(flow0, flow1, data)
    n1right = np.count_nonzero(np.logical_and((response < cut), (signal == 1)))
    n0right = np.count_nonzero(np.logical_and((response > cut), (signal == 0)))
    return (n1right + n0right)/len(signal)


print(proportionRight(tf.concat([batched_val_data0, batched_val_data1], axis = 0),np.concatenate((np.zeros(len(batched_val_data0)),np.ones(len(batched_val_data1)))), flow0, flow1))

validation0=classify(flow0, flow1, batched_val_data0)
validation1=classify(flow0, flow1, batched_val_data1)

dataResp0= classify(flow0, flow1, x_data[y_data == 0])
dataResp1= classify(flow0, flow1, x_data[y_data == 1])

#plotting
#------------------------------------------------------------------------------


#plotranges = (np.min(tf.concat([validation0, validation1], axis = 0).numpy()), np.max(tf.concat([validation0, validation1], axis = 0).numpy()))
plotranges = (-100, 100)

plt.title("validation response")
plt.hist(validation0.numpy(), range=plotranges, alpha = 0.7 ,  bins= 500, label = f"background; entries: {len(validation0)}")
plt.hist(validation1.numpy(), range=plotranges, alpha = 0.7 ,  bins= 500, label = f"signal; entries: {len(validation1)}")
plt.xlabel(r"$\frac{\ln(flow_0)}{\ln(flow_1)}$")
plt.legend()
plt.yscale("log")
plt.savefig("figs/fracln_validation_hist.png", format="png")
plt.clf()


plt.title("whole data response")
plt.hist(dataResp0.numpy(), range=plotranges, alpha = 0.7 ,  bins= 500, label = f"background; entries: {len(dataResp0)}")
plt.hist(dataResp1.numpy(), range=plotranges, alpha = 0.7 ,  bins= 500, label = f"signal; entries: {len(dataResp1)}")
plt.xlabel(r"$\frac{\ln(flow_0)}{\ln(flow_1)}$")
plt.legend()
plt.yscale("log")
plt.savefig("figs/fracln_data_hist.png", format="png")
plt.clf()

roc = roc_curve(np.concatenate((np.zeros(len(validation0)),np.ones(len(validation1)))),tf.concat([validation0, validation1], axis = 0).numpy(),pos_label=1, drop_intermediate=False)
plt.plot(roc[0],roc[1])
plt.xlabel("1 - purity eg. error rate")
plt.ylabel("efficiency")
plt.savefig("figs/ROC_validation.png", format="png")
plt.clf()

plt.title("training loss function values")
plt.plot(train_losses[0], label="flow0")
plt.plot(train_losses[1], label="flow1")
plt.xlabel("number of training steps")
plt.ylabel("training loss")
plt.legend()
plt.savefig("figs/training_loss_both.png", format="png")
plt.clf()

plt.title("training loss function values")
plt.plot(train_losses[0], label="flow0", color= "tab:blue")
plt.xlabel("number of training steps")
plt.ylabel("training loss")
plt.legend()
plt.savefig("figs/training_loss_0.png", format="png")
plt.clf()

plt.title("training loss function values")
plt.plot(train_losses[1], label="flow1", color= "tab:orange")
plt.xlabel("number of training steps")
plt.ylabel("training loss")
plt.legend()
plt.savefig("figs/training_loss_1.png", format="png")
plt.clf()