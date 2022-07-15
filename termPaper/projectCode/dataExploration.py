import astroML.datasets
import matplotlib.pyplot as plt

#load data into /tmp/astroML_data
x_data, y_data = astroML.datasets.fetch_rrlyrae_combined(data_home="/tmp/astroML_data", download_if_missing=True)

featurelabels = ["u-g","g-r", "r-i", "i-z"]
# u-g, g-r, r-i, i-z
# sind irgendwelche Filter
# u ultraviolet, g green, r red, i infrared, z nm = 900

#histogramms split by label
#------------------------------------------------------------------------------
for i in range(len(x_data[0])):
    plt.hist(x_data[y_data==0,i], bins= 100, label="background star", alpha=0.7, color = "tab:orange")
    plt.hist(x_data[y_data==1,i], bins= 100, label="RR Lyrae", alpha=1, color = "tab:blue")
    plt.yscale("log")
    plt.xlabel(f"{featurelabels[i]}") 
    plt.ylabel("Number of stars")
    plt.legend()
    plt.savefig(f"figs/hist_xData_feature{i}_labelSplit.png",format="png")
    plt.clf()

#feature histogramm
#------------------------------------------------------------------------------
for i in range(len(x_data[0])):
    plt.hist(x_data[:,i], bins= 100)
    plt.xlabel(f"{featurelabels[i]}") 
    plt.ylabel("Number of stars")
    plt.savefig(f"figs/hist_xData_feature{i}.png",format="png")
    plt.clf()

#Kolmogorov-Smirnov test
#------------------------------------------------------------------------------
#high value is indicator for good feature
from scipy.stats import ks_2samp

for i in range(len(x_data[0])):
    print(f"{featurelabels[i]}\t",ks_2samp(x_data[y_data==0,i], x_data[y_data==1,i]))

#correlation between features
#------------------------------------------------------------------------------
import itertools as it
import numpy as np


fig, ax = plt.subplots(len(x_data[0]),len(x_data[0]), figsize=(20,20))
plt.subplots_adjust(hspace=0.5) # has to be after subplots

for ij in it.product(range(len(x_data[0])), repeat=2):
    ax[ij[0]][ij[1]].hist2d(x_data[:,ij[0]],x_data[:,ij[1]], bins=100)
    ax[ij[0]][ij[1]].set_xlabel(f"{featurelabels[ij[0]]}")
    ax[ij[0]][ij[1]].set_ylabel(f"{featurelabels[ij[1]]}")


plt.savefig(f"figs/correlation_features.png",format="png")
plt.clf()

# just signal == 1
fig, ax = plt.subplots(len(x_data[0]),len(x_data[0]), figsize=(20,20))
plt.subplots_adjust(hspace=0.5) # has to be after subplots so that labels dont overlap


for ij in it.product(range(len(x_data[0])), repeat=2):
    ax[ij[0]][ij[1]].hist2d(x_data[y_data==1,ij[0]],x_data[y_data==1,ij[1]], bins=25)
    ax[ij[0]][ij[1]].set_xlabel(f"{featurelabels[ij[0]]}")
    ax[ij[0]][ij[1]].set_ylabel(f"{featurelabels[ij[1]]}")

plt.savefig(f"figs/correlation_features_label1.png",format="png")
plt.clf()