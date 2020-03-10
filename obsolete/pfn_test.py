from utils import helpers, models
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from keras import optimizers,losses,callbacks
import energyflow as ef
from energyflow.archs import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split,ptyphims_from_p4s,ms_from_p4s

# train, val, test = 75000, 10000, 15000

# X, y = qg_jets.load(train + val + test)
# inputFile="input/total_CP5.root"
inputFile=".bkp_input/input/total_CP5.root"
# inputFile="/nfs/dust/cms/user/sewuchte/analysisFWK/dnn/CMSSW_9_4_13_patch2/src/TopAnalysis/Configuration/analysis/diLeptonic/plainTree_2016/Nominal/ee/ee_ttbarsignalplustau_fromDilepton.root"
treeName="plainTree_rec_step8"
# normVal = 170.
# normVal = 340.
normVal = 1.
# xData, yData = helpers.loadData2(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)
xData, yData = helpers.loadDataNew2(inputFile, treeName,zeroJetPadding=True)
# xData, yData = helpers.loadData2(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)
x_, x_test, y_, y_test = train_test_split(xData, yData, test_size=0.33)
x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.33)



x_train = x_train.reshape((x_train.shape[0],38*4))
x_test = x_test.reshape((x_test.shape[0],38*4))
x_val = x_val.reshape((x_val.shape[0],38*4))

# from sklearn.preprocessing import StandardScaler
# scaler_x = StandardScaler()
# scaler_x.fit(x_train)

# x_train = scaler_x.transform(x_train)
# x_test = scaler_x.transform(x_test)
# x_val = scaler_x.transform(x_val)

x_train=x_train/340.
x_test=x_test/340.
x_val=x_val/340.

x_train = x_train.reshape((x_train.shape[0],38,4))
x_test = x_test.reshape((x_test.shape[0],38,4))
x_val = x_val.reshape((x_val.shape[0],38,4))

from sklearn.preprocessing import StandardScaler
scaler_y = StandardScaler()
scaler_y.fit(y_train)

# print (y_train)
# print (y_train.shape)
# print(scaler_y.mean_)

y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)
y_val = scaler_y.transform(y_val)
# print (y_train)
# print (y_train.shape)



Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
# Phi_sizes, F_sizes = (10, 10, 10,10,10,64), (64, 128, 128,128,64)
# Phi_sizes, F_sizes = (200, 200, 512), (200, 200, 200)
def custom_loss(y_true, y_pred):
    m_true = tf.math.reduce_mean(y_true,1,keep_dims=True)
    m_pred = tf.math.reduce_mean(y_pred,1,keep_dims=True)
    s_true = tf.math.reduce_std(y_true,1,keepdims=True)
    s_pred = tf.math.reduce_std(y_pred,1,keepdims=True)
    d_true=tf.divide(tf.subtract(y_true,m_true),s_true)
    d_pred=tf.divide(tf.subtract(y_pred,m_pred),s_pred)
    loss = losses.mean_squared_error(d_true,d_pred)
    return loss


# print (x_train)
# print (x_train.shape)
# print (x_train.shape[-1])
# print (y_train)
# print (y_train.shape)
# x_train2 = m.evaluate(x_train)
# y_train2 = m.evaluate(y_train)
# optim = optimizers.Adam(lr=0.001)
optim = optimizers.Adam(lr=0.00001)
# optim = optimizers.Adam(lr=0.00001)
# pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'],output_act='linear')
# pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'],output_act='linear',
#             Phi_acts='softplus',F_acts='softplus')
# pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss=custom_loss, metrics=['mean_squared_error','mean_absolute_error'],output_act='linear',
#             Phi_acts='relu',F_acts='relu')
# pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss="mean_squared_error", metrics=['mean_squared_error','mean_absolute_error'],output_act='linear',
#             Phi_acts='relu',F_acts='relu',patience=5)
# pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'],output_act='linear',
#         optimizer=optim)
# pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'],output_act='linear',
#             Phi_acts='selu',F_acts='selu',latent_dropout=0.15,F_dropouts=0.15,Phi_l2_regs=1e-7,F_l2_regs=1e-7,optimizer=optim)
# pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'],output_act='linear',
#             Phi_acts='softplus',F_acts='softplus',latent_dropout=0.1,F_dropouts=0.1,optimizer=optim)




from functools import partial
#~ import tensorflow as tf

slim = tf.contrib.slim


################################################################################
# SIMILARITY LOSS
################################################################################
def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.
  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  # if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
  #   raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
  r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
  We create a sum of multiple gaussian kernels each having a width sigma_i.
  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))






def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  print("shapes",x.shape,y.shape)
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


# def mmd_loss(source_samples, target_samples, weight, scope=None):
# def mmd_loss(source_samples, target_samples):
def mmd_loss(y_true, y_pred):
  """Adds a similarity loss term, the MMD between two representations.
  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.
  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    # scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      y_true, y_pred, kernel=gaussian_kernel)
      # source_samples, target_samples, kernel=gaussian_kernel)
  # loss_value = tf.maximum(1e-4, loss_value) * weight
  loss_value = tf.maximum(1e-4, loss_value)
  assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
  with tf.control_dependencies([assert_op]):
    tag = 'MMD Loss'
    # if scope:
    #   tag = scope + tag
    tf.summary.scalar(tag, loss_value)
    tf.losses.add_loss(loss_value)

  return loss_value





# pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss="mean_squared_error", metrics=['mean_squared_error','mean_absolute_error'],output_act='linear',
#             Phi_acts='relu',F_acts='relu',patience=5)
pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss=mmd_loss, metrics=['mean_squared_error','mean_absolute_error'],output_act='linear',
            Phi_acts='relu',F_acts='relu',patience=5)
# pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4, loss="mean_squared_error",
#             metrics=['mean_squared_error','mean_absolute_error'], output_act='linear',
#             optimizer=optim)


reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3)
earlyStop = callbacks.EarlyStopping(monitor='val_loss',patience=4)

pfn.fit(x_train, y_train,
          epochs=150,
          batch_size=1024,
          validation_data=(x_val, y_val),
          verbose=1,
          callbacks=[earlyStop,reduce_lr])



# y_predicted_norm = pfn.predict(x_test)
# y_predicted = helpers.normalizeData(y_predicted_norm, normVal, invert=True)
# y_test_nonNorm = helpers.normalizeData(y_test, normVal, invert=True)
y_predicted_norm = pfn.predict(x_test)
y_predicted = scaler_y.inverse_transform(y_predicted_norm)
y_test_nonNorm = scaler_y.inverse_transform(y_test)

# y_predicted_norm = pfn.predict(x_test)
# y_predicted = y_predicted_norm
# y_test_nonNorm = y_test

# y_predicted = scalerY.inverse_transform(y_predicted_norm)
# y_test_nonNorm = scalerY.inverse_transform(y_test)
helpers.doPredictionPlots(y_test_nonNorm, y_predicted, "output/pfn/")
# helpers.doPredictionPlots2(y_test_nonNorm, y_predicted, "output/pfn/")
# helpers.doPredictionPlots2(y_test_nonNorm, y_predicted, "output/pfn/")
# helpers.doPredictionPlots(y_test, y_predicted, "output/pfn/")
