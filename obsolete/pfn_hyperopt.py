from __future__ import print_function
from utils import helpers
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe

from hyperas import optim
from hyperas.distributions import choice, uniform

# from utils import helpers, models
from sklearn.model_selection import train_test_split

# import numpy as np
# import tensorflow as tf
# from keras import optimizers
# import energyflow as ef
from energyflow.archs import PFN



def data():
    inputFile = "input/total_CP5.root"
    treeName = "plainTree_rec_step8"
    normVal = 1.
    xData, yData = helpers.loadData3(inputFile, treeName, loadGenInfo=False,
                zeroJetPadding=False, normX=False, normY=False, normValue=normVal)
    frac = int(0.15*len(xData))
    xData = xData[:frac:]
    yData = yData[:frac:]
    x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.35)
    # x_, x_test, y_, y_test = train_test_split(xData, yData, test_size=0.33)
    # x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.33)
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
    Phi_sizes = ()
    F_sizes = ()
    #
    nP = {{choice([2, 4, 6])}}
    nF = {{choice([3, 5, 7])}}
    Pdim = {{choice([100, 200, 500])}}
    POutdim = {{choice([20, 100, 250])}}
    Fdim = {{choice([100, 200, 500])}}
    for i in range(nF-1):
        # Phi_sizes.append(Pdim)
        Phi_sizes+=(Pdim,)
    Phi_sizes+=(POutdim,)
    # for i in range(nF):
    #     Phi_sizes+=(Pdim,)
    # # Phi_sizes.append(POutdim)
    for i in range(nF):
        F_sizes+=(Fdim,)

    acts={{choice(['relu','selu','softplus'])}}

    # pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4,
    #             loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'],
    #             output_act='linear', Phi_acts={{choice(['relu','softplus'])}},F_acts={{choice(['relu','softplus'])}},
    #             latent_dropout={{uniform(0, 1)}},F_dropouts={{uniform(0, 1)}}, optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
    pfn = PFN(input_dim=x_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=4,
                loss='mean_squared_error', metrics=['mean_squared_error','mean_absolute_error'],
                output_act='linear',Phi_acts=acts,F_acts=acts,latent_dropout={{uniform(0, 1)}},F_dropouts={{uniform(0, 1)}},patience=5)

    result = pfn.fit(x_train, y_train,
              epochs=15,
              batch_size={{choice([64, 128, 256, 512, 1024])}},
              validation_split=0.1,
              verbose=2)

    validation_loss = np.amin(result.history['val_loss'])
    print('Best validation loss of epoch:', validation_loss)
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': pfn}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())
    # X_train, Y_train, X_test, Y_test = data()
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
