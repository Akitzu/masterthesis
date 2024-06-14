"""Useful tools for looking at connection length data in 
a single (constant phi) slice."""

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import re
from matplotlib import cm

class LcVolume():


	def __init__(self, filename):
		""" Initialise the instance and extract data from file."""
		self.filename = filename 
		self.extract_data()
		return 

	def extract_data(self, scaling=1e-2):
		"""Read a volume lc file and return
		data = [R_longarray, Z_longarray, lc_longarray]
		'longarray' means a 1D array.
		The data in the file is logically rectanglar in radial and 
		poloidal coordinates, but with some "missing"
		entries (non-convex cells are not simulated).
		The data in the file contains, in this order:
		1) header information
		2) R coordinates of cell corners
		3) Z coordinates of cell corners
		4) For 
		 """

		zero_tol = 1E-6
		myfile = open(self.filename, "r")
		filetext = myfile.read()
		myfile.close()

		lines = re.split("\n", filetext.strip())
		info_line=lines[0].strip()
		info_elems = re.split("\s+", info_line)
		nrad = int(info_elems[0]) ; npol = int(info_elems[1]) 
		phi = float(info_elems[0])

		RZlc_list = []

		for line in lines[1:]:
			line_elems = re.split("\s+", line.strip())
			for RZlc_string in line_elems:
				RZlc_list.append(float(RZlc_string))
		# Now split based on nrad, npol
		npoints = nrad*npol	# points = cell corners
		Rvals = RZlc_list[:npoints]
		Zvals = RZlc_list[npoints:2*npoints]
		lcvals = RZlc_list[2*npoints:]

		## no. cells is less than no. cell corners by 1 in 
		## each dimension.
		R_cell_centre_longarray = np.zeros(((nrad-1)*(npol-1)))
		Z_cell_centre_longarray = np.zeros(((nrad-1)*(npol-1)))
		lc_cell_centre_longarray = np.zeros(((nrad-1)*(npol-1)))

		## arrays to store the data
		self.R_cell_corner_mesh = np.zeros((nrad, npol))
		self.Z_cell_corner_mesh = np.zeros((nrad, npol))
		self.R_cell_centre_mesh = np.zeros((nrad-1, npol-1))
		self.Z_cell_centre_mesh = np.zeros((nrad-1, npol-1))
		self.lc_cell_centre_mesh = np.zeros((nrad-1, npol-1))
		self.nrad = nrad
		self.npol = npol
		for pol_idx in range(0, npol):
			for R_idx in range(0, nrad):
				compound_idx = pol_idx*(nrad) + R_idx
				self.R_cell_corner_mesh[R_idx, pol_idx] = Rvals[compound_idx] * scaling
				self.Z_cell_corner_mesh[R_idx, pol_idx] = Zvals[compound_idx] * scaling
		for pol_idx in range(0, npol-1):
			for R_idx in range(0, nrad-1):
				cell_idx = pol_idx*(nrad-1) + R_idx
				lR_lpol_idx = pol_idx*nrad + R_idx
				uR_lpol_idx = lR_lpol_idx + 1
				lR_upol_idx = (pol_idx+1)*nrad + R_idx
				uR_upol_idx = lR_upol_idx + 1
				Rval = 0.25*(Rvals[lR_lpol_idx]
									+ Rvals[uR_lpol_idx]
									+ Rvals[lR_upol_idx]
									+ Rvals[uR_upol_idx])
				Zval = 0.25*(Zvals[lR_lpol_idx]
									+ Zvals[uR_lpol_idx]
									+ Zvals[lR_upol_idx]
									+ Zvals[uR_upol_idx])

				lcval = lcvals[cell_idx]
				self.lc_cell_centre_mesh[R_idx, pol_idx] = lcval	
				self.R_cell_centre_mesh[R_idx, pol_idx] = Rval * scaling
				self.Z_cell_centre_mesh[R_idx, pol_idx] = Zval * scaling
				R_cell_centre_longarray[cell_idx] = Rval 	
				Z_cell_centre_longarray[cell_idx] = Zval  
				lc_cell_centre_longarray[cell_idx] = lcval

				### If the cell is non-convex, EMC3-Lite sets the 
				### R and Z coordinates of the relevant cell corners
				### to zero. Since we never expect R=0 data for realistic
				### stellarators, we can check if a cell is "good"
				### by checking if |R|>0 for all cell corners.
				### If bad, set the data to nan. 
				if not ((abs(Rvals[lR_lpol_idx]) > zero_tol) and
						(abs(Rvals[lR_upol_idx]) > zero_tol) and
						(abs(Rvals[uR_lpol_idx]) > zero_tol) and
						(abs(Rvals[uR_upol_idx]) > zero_tol)):

					R_cell_centre_longarray[cell_idx] = np.NaN			
					Z_cell_centre_longarray[cell_idx] = np.NaN			
					lc_cell_centre_longarray[cell_idx] = np.NaN			
					self.R_cell_centre_mesh[R_idx, pol_idx] = np.NaN	
					self.Z_cell_centre_mesh[R_idx, pol_idx] = np.NaN	
					self.lc_cell_centre_mesh[R_idx, pol_idx] = np.NaN	

		# Get the finite values of the longarrays only.
		finite_idxs = np.isfinite(R_cell_centre_longarray)
		self.R_cell_centre_longarray = R_cell_centre_longarray[finite_idxs] * scaling
		self.Z_cell_centre_longarray = Z_cell_centre_longarray[finite_idxs] * scaling
		self.lc_cell_centre_longarray = lc_cell_centre_longarray[finite_idxs]

		return

	def make_lc_volume_plot(self, fig, ax1,
		cbar_max=None,
		nlevels = 100,
		ax_title_fontsize=12,
		logscale=False,
		cmap_str="viridis",
		which_plotting_option = "tricontourf",
		scatter_marker_size = 5,
		scatter_ms = "s",
		min_val_linear=0,
		min_val_log=1,
		):
		"""Put the data on a plot. The main parameters to 
		think about are:
		1) colorscale: linear or log, and whether to set cutoffs
		2) which_plotting_option. Recommend 'fill' for the prettiest 
		and most accurate representation of the data. But 'scatter' and 'tricontourf'
		are much faster. """

		def prettify_plot():
			if logscale:
				if abs(max_val-5) < 1e-4:
					#cbar.set_ticks(np.linspace(min_val, max_val, 7))
					#cbar.set_ticklabels([r"$1$", r"$10$", r"$10^2$", r"$10^3$", r"$10^3$", r"$10^4$",r"$10^5$"])
					cbar.set_ticks(np.linspace(min_val, max_val, 6))
					cbar.set_ticklabels([r"$10$", r"$10^2$", r"$10^3$", r"$10^3$", r"$10^4$",r"$10^5$"])
#				else:
#				cbar.set_ticks(np.linspace(min_val, max_val, 6))					
			else:
				cbar.set_ticks(np.linspace(min_val, max_val, 6))
			cbax1.set_title(r"$L_C$" +"\n" + r"(cm)", fontsize=ax_title_fontsize)

		def get_cbar_data():
			"""Get the colormap, extend and max/min vals"""
			cmap = cm.get_cmap(cmap_str)
			
			if cbar_max is not None:
				max_val = cbar_max 
			else:
				finite_idxs = np.isfinite(self.lc_cell_centre_mesh)
				max_val = np.max(self.lc_cell_centre_mesh[finite_idxs])

			if logscale:
				extend="both"
				min_val = min_val_log
				#min_val = 10
				max_val = np.log10(max_val)
			else:
				extend="max"
				min_val = min_val_linear

			return min_val, max_val, cmap, extend

		inner_R = self.R_cell_corner_mesh[0,:]
		inner_Z = self.Z_cell_corner_mesh[0,:]
		finite_idxs = np.isfinite(inner_R)
		inner_R = inner_R[finite_idxs]
		inner_Z = inner_Z[finite_idxs]
		inner_lc = np.ones((len(inner_R)))*-20
		new_R_longarray = np.concatenate((self.R_cell_centre_longarray, inner_R))
		new_Z_longarray = np.concatenate((self.Z_cell_centre_longarray, inner_Z))
		new_lc_longarray = np.concatenate((self.lc_cell_centre_longarray, inner_lc))

		min_val, max_val, cmap, extend = get_cbar_data()
		levels = np.linspace(min_val, max_val, nlevels)
		print("max_val = ", max_val)
		#sys.exit()
		if logscale:
			lc_longarray = np.log10(self.lc_cell_centre_longarray)
			lc_longarray[np.logical_not(np.isfinite(lc_longarray))] = 1
			lc_cell_centre_mesh = np.log10(self.lc_cell_centre_mesh)
			lc_cell_centre_mesh[np.logical_not(np.isfinite(lc_cell_centre_mesh))] = 1
		else:
			lc_longarray = self.lc_cell_centre_longarray
			lc_cell_centre_mesh = self.lc_cell_centre_mesh

		if which_plotting_option == "tricontourf":
			cf = ax1.tricontourf(self.R_cell_centre_longarray, 
			 	self.Z_cell_centre_longarray, lc_longarray,
			 	levels=levels, cmap=cmap, extend=extend)
		elif which_plotting_option == "scatter":
			cf = ax1.scatter(self.R_cell_centre_longarray, 
			 	self.Z_cell_centre_longarray, c=lc_longarray,
				cmap=cmap, s=scatter_marker_size, marker=scatter_ms, vmin=min_val,
				vmax=max_val)
			ax1.set_facecolor("grey")
		elif which_plotting_option == "fill":
			for rad_idx in range(0, self.nrad-1):
				print("rad_idx = {:}/{:}".format(rad_idx,self.nrad-2))
				for pol_idx in range(0, self.npol-1):
					## Check Lc is not nan
					lcval = lc_cell_centre_mesh[rad_idx, pol_idx]
					if np.isfinite(lcval):
						## Get the color
						col= cmap((lcval-min_val)/(max_val-min_val))
						## Get the corners
						R_ll = self.R_cell_corner_mesh[rad_idx, pol_idx]
						R_lu = self.R_cell_corner_mesh[rad_idx, pol_idx+1]
						R_ul = self.R_cell_corner_mesh[rad_idx+1, pol_idx]
						R_uu = self.R_cell_corner_mesh[rad_idx+1, pol_idx+1]
						Z_ll = self.Z_cell_corner_mesh[rad_idx, pol_idx]
						Z_lu = self.Z_cell_corner_mesh[rad_idx, pol_idx+1]
						Z_ul = self.Z_cell_corner_mesh[rad_idx+1, pol_idx]
						Z_uu = self.Z_cell_corner_mesh[rad_idx+1, pol_idx+1]
						## Fill
						## Don*t fill if any Rval is zero
						if (R_ll * R_lu * R_ul * R_uu) > 1e-5:
							ax1.fill([R_ll, R_lu, R_uu, R_ul], [Z_ll, Z_lu, Z_uu, Z_ul], c=col)
			cf = ax1.scatter([0,0], [0,0], c=[0,1],
				cmap=cmap, s=scatter_marker_size, marker=scatter_ms, vmin=min_val,
				vmax=max_val)
		ax1.plot(inner_R, inner_Z, lw=2, c="black")
		# cbar = fig.colorbar(cf, cax=cbax1)
		cbar = fig.colorbar(cf)
		cbax1 = cbar.ax

		prettify_plot()

		return cbar


