import numpy as np
import pathlib
import pandas as pd


def read_students():
    xvals_train = pd.read_csv('x_train.csv').to_numpy()
    yvals_train = pd.read_csv('y_train.csv').to_numpy()
    
    xvals_test = pd.read_csv('x_test.csv').to_numpy()
    yvals_test = pd.read_csv('y_test.csv').to_numpy()
    return xvals_train,yvals_train,xvals_test,yvals_test


