"""A minimal working example for plotting the connection 
length data from the GYM configuration """

import matplotlib.pyplot as plt
import numpy as np
from LcVolume import LcVolume

### Data
lcdata_phi0_filename = "LC_phi0_all_targets.txt"
lcdata_phi12_filename = "LC_phi12_all_targets.txt"
lcdata_phi18_filename = "LC_phi18_all_targets.txt"
lcdata_phi36_filename = "LC_phi36_all_targets.txt"

def make_plot_phi0(ax, filename, tpl=(0.9, 0.1, 0.03, 0.8)):
	""" """
	## Load data
	lcdata_phi0 = LcVolume(filename)

	fig = ax.get_figure()
	### Change which_plotting_option to "scatter" or "tricontourf"
	### for faster plotting.
	bar = lcdata_phi0.make_lc_volume_plot(fig, ax, which_plotting_option="fill")
	fig.add_axes(bar.ax)

	phi_val=0
	fig_title=r"$\phi=$" + "{:d}°".format(phi_val)
	ax.set_facecolor('gray')
	ax.set_xlabel("R (cm)", fontsize=14)
	ax.set_ylabel("Z (cm)", fontsize=14)		
	ax.set_title(fig_title)

if __name__ == "__main__":
	make_plot_phi0()
