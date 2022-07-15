import astroML.datasets
import matplotlib.pyplot as plt

x_data, y_data = astroML.datasets.fetch_rrlyrae_combined(data_home="/tmp/astroML_data", download_if_missing=True)

#histogramms split by label
#------------------------------------------------------------------------------
for i in range(len(x_data[0])):
    plt.hist(x_data[y_data==0,i], bins= 100, label="background star", alpha=0.7, color = "tab:orange")
    plt.hist(x_data[y_data==1,i], bins= 100, label="RR Lyrae", alpha=1, color = "tab:blue")
    plt.yscale("log")
    plt.xlabel(f"feature {i}") #todo featureliste
    plt.ylabel("Number of stars")
    plt.legend()
    plt.savefig(f"figs/hist_xData_feature{i}_labelSplit.png",format="png")
    plt.clf()

#feature histogramm
#------------------------------------------------------------------------------
for i in range(len(x_data[0])):
    plt.hist(x_data[:,i], bins= 100)
    plt.xlabel(f"feature {i}") #todo featureliste
    plt.ylabel("Number of stars")
    plt.savefig(f"figs/hist_xData_feature{i}.png",format="png")
    plt.clf()

#Kolmogorov-Smirnov test
#------------------------------------------------------------------------------
#high value is indicator for good feature
from scipy.stats import ks_2samp

for i in range(len(x_data[0])):
    print(f"feature {i}\t",ks_2samp(x_data[y_data==0,i], x_data[y_data==1,i]))




#todo correlation plots in 4 x 4 grid