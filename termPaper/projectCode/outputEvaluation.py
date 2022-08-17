# plotting of data form varOut/parameterVariationOut.csv

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

az_path = "./varOut/"
az_name = "parameterVariationOut.csv"

data = pd.read_csv(az_path+az_name, sep=",", header=None)

default = [1000.0,100.0,0.0,200.0,2.0,100.0,2.0,0.001,0.0001,12.0]


data["combined"] = data[[i for i in range(len(default))]].values.tolist()


print(data[[default == row for row in data['combined']]].iloc[:,10])



plt.title("validation correct classified for default parameter")
plt.hist(data[[default == row for row in data['combined']]].iloc[:,10])
plt.xlabel(r"proportion of correct classified ")
plt.legend()
plt.savefig("figs/hist_corrcl.png", format="png")
plt.clf()