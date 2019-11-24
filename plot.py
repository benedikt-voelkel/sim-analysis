import ROOT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import os
import numpy as np
import argparse
from math import sqrt
import sys
np.set_printoptions(threshold=sys.maxsize)

from plot_utils.plot_config import PlotConfig
from plot_utils.data_manager import PlotManager
from plot_utils.logger import configure_logger

PARTICLE_MAP = {"neutron": "n",
                "proton": "p",
                "e-": "$e^-$",
                "e+": "$e^+$",
                "mu-": "$\mu^-$",
                "mu+": "$\mu^+$",
                "pi-": "$\pi^-$",
                "pi+": "$\pi^+$",
                "K-": "$\mathrm{K}^-$",
                "K+": "$\mathrm{K}^+$",
                "gamma": "$\gamma$",
                "50000050": "Cerenkov $\gamma$"
                }

def replace_tex(label):
    return "$" + label + "$"

def make_patch_spines_invisible(ax):
	ax.set_frame_on(True)
	ax.patch.set_visible(False)
	for sp in ax.spines.values():
		sp.set_visible(False)


"""
Parse and handle arguments and dispatch functions
"""
parser = argparse.ArgumentParser()
parser.add_argument("--plot-config", "-c", dest="plot_config", help="plot configuration file", type=str, required=True)

args = parser.parse_args()

configure_logger(False)

config = PlotConfig(args.plot_config)

plt_mgr = PlotManager(config)

plt_mgr.plot_all()
