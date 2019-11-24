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

def add_to_dict(filename, histo_name, dict_extend):

    print(f"Processing histogram {histo_name} in file {filename}")
    root_file = ROOT.TFile.Open(filename, "READ")
    root_histo_dir = root_file.Get("MCAnalysisObjects")
    histo = root_histo_dir.Get(histo_name)

    if isinstance(histo, ROOT.TH2):
        xAxis = histo.GetXaxis()
        yAxis = histo.GetYaxis()
        for ibinX in range(1, histo.GetNbinsX() + 1):
            for ibinY in range(1, histo.GetNbinsY() + 1):
                labelX = xAxis.GetBinLabel(ibinX)
                labelY = yAxis.GetBinLabel(ibinY)
                if labelX == "" or labelY == "":
                    continue
                content = histo.GetBinContent(ibinX, ibinY)

                if labelY not in dict_extend:
                    dict_extend[labelY] = {}
                if labelX not in dict_extend[labelY]:
                    dict_extend[labelY][labelX] = 0
                dict_extend[labelY][labelX] += content
    else:
        axis = histo.GetXaxis()
        for ibin in range(1, histo.GetNbinsX() + 1):
            label = axis.GetBinLabel(ibin)
            if label == "":
                continue
            if label not in dict_extend:
                dict_extend[label] = [0, 0.]
            content = histo.GetBinContent(ibin)
            error = histo.GetBinError(ibin)
            dict_extend[label][0] += content
            dict_extend[label][1] = sqrt( dict_extend[label][1]**2 + error**2 )

def pad_dicts(dicts_to_pad, pad_with):
    all_labels = []
    for d in dicts_to_pad:
        for l in d.keys():
            if l not in all_labels:
                all_labels.append(l)
    for l in all_labels:
        for d in dicts_to_pad:
            if l not in d:
                d[l] = pad_with


def plot_1d(to_plot, yerr=None, x_ticks_labels=None, x_axis_title=None, y_axis_titles=None, output_filename="plot", ):


	colors = ["r", "b", "g"]
	markers = ["P", "X", "v"]

	# Fix width of entire figure
	figure_width = 40
	# fix graph height
	figure_height = 30

	dpi = 100

	global_font_size = 50

	fig = plt.figure(tight_layout=True, figsize=(figure_width, figure_height), dpi = dpi)

	if y_axis_titles is None:
		y_axis_titles = [ f"axis{i}" for i in range(len(to_plot)) ]

	elif len(to_plot) != len(y_axis_titles):
		print(f"ERROR: {len(to_plot)} plots but {len(y_axis_titles)} axis titles")
		exit(1)

	x_axis_title = x_axis_title if x_axis_title is not None else ""

	ax_main = None
	for i in range(len(to_plot)):
		ax = None
		if ax_main is None:
			ax_main = fig.add_subplot()
			ax = ax_main
			#ax_main.errorbar(np.arange(len(to_plot[i])), to_plot[i], yerr=yerr[i], color=colors[i%len(colors)])
			ax_main.set_xticks(np.arange(len(x_ticks_labels)))
			x_ticks_labels = [ l if l not in PARTICLE_MAP else PARTICLE_MAP[l] for l in x_ticks_labels ]
			ax_main.set_xticklabels(x_ticks_labels, fontsize=global_font_size)
			ax_main.set_xlabel(x_axis_title, fontsize=global_font_size)
			#ax_main.tick_params('y', colors=colors[i%len(colors)], labelsize=global_font_size)
			#ax_main.set_ylabel(axis_titles[i], color=colors[i%len(colors)], fontsize=global_font_size)
			plt.setp(ax_main.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
		else:
			ax = ax_main.twinx()
		#ax.errorbar(np.arange(len(to_plot[i])), to_plot[i], yerr=yerr[i], color=colors[i%len(colors)], ls="None")
		ax.scatter(np.arange(len(to_plot[i])), to_plot[i], color=colors[i%len(colors)], marker=markers[i%len(markers)], s=dpi *14, linewidths=0.5)
		ax.tick_params('y', colors=colors[i%len(colors)], labelsize=global_font_size)
		ax.set_ylabel(y_axis_titles[i], color=colors[i%len(colors)], fontsize=global_font_size)
		# TODO That can be made better and flexible, but for noww...
		if i > 1:
			# Offset axis
			ax.spines["right"].set_position(("axes", 1.1))
			# Make spines and patch invisible
			make_patch_spines_invisible(ax)
			# Second, show the right spine.
			ax.spines["right"].set_visible(True)

	plt.savefig(output_filename + ".eps")
	plt.savefig(output_filename + ".png")

"""
Parse and handle arguments and dispatch functions
"""
parser = argparse.ArgumentParser()
parser.add_argument("--ylabels", "-y", help="y-labels", action="append", required=True)
parser.add_argument("--normalize", help="flag enabling normalization", action="store_true")
parser.add_argument("--xlabel", "-x", help="x-label", default="")
parser.add_argument("--ratio", action="store_true")
parser.add_argument("--analysis-files", "-f", dest="analysis_files", help="files containing histograms", nargs="+")
parser.add_argument("--histo-name", "-n", dest="histo_name", help="histogram name", type=str, action="append")
parser.add_argument("--axis-group", "-g", dest="axis_group", help="axis group sharing one axis", type=int, action="append", default=1)
parser.add_argument("--include-bins", dest="include_bins", help="only selected bins will be shown", nargs="+")
parser.add_argument("--output-filename", "-o", dest="output_filename", help="file where to store the plot", default="plot")
parser.add_argument("--n-xbins", dest="n_xbins", help="restrict number of x-bins", type=int, default=0)
parser.add_argument("--dimension", "-d", help="expected histogram dimension", type=int, default=1)


parser.add_argument("--plot-config", "-d", help="expected histogram dimension", type=int, default=1)




args = parser.parse_args()

times_dict = {}
steps_dict = {}

if len(args.ylabels) != len(args.histo_name):
	print(f"ERROR: need exactly as many ylabels as histogram groups")
	exit(1)


if args.dimension == 1:
    for f in args.analysis_files:
        add_to_dict(f, args.histo_name[0], times_dict)
        add_to_dict(f, args.histo_name[1], steps_dict)

    pad_dicts([times_dict, steps_dict], [0, 0])

    labels = [ k for k in times_dict ]
    print(labels)
    if args.include_bins is not None:
        labels = [ l for l in labels if l in args.include_bins ]

    print(labels)
    labels_to_index = { labels[i]: i for i in range(len(labels)) }

	# Prepare the lists so that there is at least a 0
    times = [ 0 for i in range(len(labels)) ]
    steps = [ 0 for i in range(len(labels)) ]
    times_err = [ 0 for i in range(len(labels)) ]
    steps_err = [ 0 for i in range(len(labels)) ]
    print(steps_dict)

    for i in range(len(labels)):
        times[i] = times_dict[labels[i]][0]
        steps[i] = steps_dict[labels[i]][0]
        times_err[i] = times_dict[labels[i]][1]
        steps_err[i] = steps_dict[labels[i]][1]
        #times_means[i] = np.mean(times_dict[labels[i]])
        #steps_means[i] = np.mean(steps_dict[labels[i]])
        #times_stds[i] = np.std(times_dict[labels[i]])
        #steps_stds[i] = np.std(steps_dict[labels[i]])

    if args.normalize is True:
        times_err = [ t / sum(times) for t in times_err]
        times = [ t / sum(times) for t in times]
        steps_err = [ t / sum(steps) for t in steps_err]
        steps = [ t / sum(steps) for t in steps]

    # Find the 25 most time consuming ones
    if len(times) > args.n_xbins and args.n_xbins > 0 :
        times = np.array(times)
        indices = times.argsort()[-args.n_xbins:][::-1]
        times = np.array(times)[indices]
        steps = np.array(steps)[indices]
        times_err = np.array(times_err)[indices]
        steps_err = np.array(steps_err)[indices]
        labels = np.array(labels)[indices]

    y_values = [ times, steps ]
    y_errors = [ times_err, steps_err ]
    y_axis_titles = [ r'' + l for l in args.ylabels ]

    if args.ratio:
        y_axis_titles.append(args.ylabels[1] + " / " + args.ylabels[0] + " [a.u.]")
        steps_per_time = [ steps[i] / float(times[i]) if times[i] > 0 else 0 for i in range(len(labels)) ]
        steps_per_time_err = [ sqrt( (steps_err[i] / float(times[i]))**2 + (steps[i] / float(times[i]) / float(times[i]) * times_err[i])**2 ) if times[i] > 0 else 0 for i in range(len(labels)) ]
        y_values.append(steps_per_time)
        y_errors.append(steps_per_time_err)

    plot_1d(y_values, yerr=y_errors, x_ticks_labels=labels, x_axis_title=args.xlabel, y_axis_titles=y_axis_titles, output_filename=args.output_filename)

elif args.dimension == 2:

    labels_x = []
    labels_y = []

    dict_2d = {}

    for f in args.analysis_files:
        add_to_dict(f, args.histo_name[0], dict_2d)

    for k1, v1 in dict_2d.items():
        for k2 in v1.keys():
            if k1 not in labels_x:
                labels_x.append(k1)
            if k2 not in labels_y:
                labels_y.append(k2)

    labels_to_index_x = { labels_x[i]: i for i in range(len(labels_x)) }
    labels_to_index_y = { labels_y[i]: i for i in range(len(labels_y)) }

    heatmap_array = np.zeros(len(labels_x) * len(labels_y)).reshape(len(labels_x), len(labels_y))

    for k1, v1 in dict_2d.items():
        for k2, v2 in v1.items():
            heatmap_array[labels_to_index_x[k1],labels_to_index_y[k2]] = v2

    max_pdg_cols = np.sum(heatmap_array, axis=1)
    #print(max_pdg_cols)
    #print(heatmap_array.shape)
    #print(heatmap_array[0,1])
    #print(heatmap_array[0,2])
    #print(heatmap_array)
    indices = max_pdg_cols.argsort()[-12:][::-1]
    #print(max_pdg_cols[indices])
    #print(indices)
    heatmap_array = heatmap_array[indices,:]
    #print(heatmap_array)
    #print(heatmap_array.shape)
    #print(heatmap_array[0,1])
    #print(heatmap_array[0,2])
    #print(heatmap_array)
    max_pdg_cols = np.sum(heatmap_array, axis=1)
    #print(max_pdg_cols)
    labels_x = np.array(labels_x)[indices]
    cbar_label = args.ylabels[0]
    # normalize
    if args.normalize:
        heatmap_array /= heatmap_array.sum()

    # Fix width of entire figure
    figure_width = 25
    # fix graph height
    figure_height = 30
    
    dpi = 50

    global_font_size = 50

    fig = plt.figure(tight_layout=True, figsize=(figure_width, figure_height), dpi = dpi)

    ax = fig.add_subplot()

    im = ax.imshow(np.transpose(heatmap_array), cmap="Reds")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=global_font_size)
    cbar.ax.tick_params(labelsize=global_font_size)


    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_yticks(np.arange(len(labels_y)))

	# Replace with beautiful labels if possible
    labels_x = [ replace_tex(l) if l not in PARTICLE_MAP else PARTICLE_MAP[l] for l in labels_x ]
    print(labels_x)

    ax.set_xticklabels(labels_x, fontsize=global_font_size)
    ax.set_yticklabels(labels_y, fontsize=global_font_size)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=global_font_size)

    # Color most important labels
    max_pdg_cols = np.sum(heatmap_array, axis=1)
    indices = max_pdg_cols.argsort()[-4:][::-1]
    x_axis_labels = ax.get_xaxis().get_ticklabels()
    for i,l in enumerate(x_axis_labels):
        if i not in indices:
            continue
        l.set_color("r")
        l.set_fontweight("extra bold")

    max_pdg_cols = np.sum(heatmap_array, axis=0)
    indices = max_pdg_cols.argsort()[-3:][::-1]
    y_axis_labels = ax.get_yaxis().get_ticklabels()
    for i,l in enumerate(y_axis_labels):
        if i not in indices:
            continue
        l.set_color("r")
        l.set_fontweight("extra bold")

    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(len(labels_x)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(labels_y)+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    plt.savefig(args.output_filename + ".eps")
    plt.savefig(args.output_filename + ".png")
