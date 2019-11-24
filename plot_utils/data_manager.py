from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from plot_utils.data_sources import DIGEST_METHODS
from plot_utils.logger import get_logger

class DataManager:
    def __init__(self, config):
        
        self.logger = get_logger()

        self.config = config
        self.data = {}

        self.read()


    def read(self):
        sources = self.config.get_sources()
        for s in sources:
            self.data[s["id"]] = DIGEST_METHODS[s["type"]](**s["args"])


    def get_data(self, data_id):
        if data_id not in self.data:
            self.logger.fatal("Unknown data id %s", str(data_id))
        return self.data[data_id]


class PlotManager:
    def __init__(self, config):
        self.data_manager = DataManager(config)
        self.config = config

        self.logger = get_logger()


        # TODO Put defaults in a dictionary
        self.default_font_size = 50
        self.default_figure_width = 40
        self.default_figure_height = 30
        self.default_dpi = 100

        self.default_x_axis_label = "x_axis"
        self.default_y_axis_label = "y_axis"

    def plot_1d(self, name, plot_config):
        custom = plot_config.get("custom", {})

        dpi = custom.get("dpi", self.default_dpi)

        fig = plt.figure(tight_layout=True,
                         figsize=(custom.get("width", self.default_figure_height), custom.get("height", self.default_figure_height)),
                         dpi=dpi)

        ax_main = None
        has_xticks = False
        for y_axis in plot_config["y_axes"]:
            # TODO Make sure sources names appear only once
            ax = None
            if ax_main is None:
                ax_main = fig.add_subplot()
                ax = ax_main
                # TODO Set x-ticks which might be different given different source. So we need to aggregate all of them first
                for s in y_axis["sources"]:
                    data = self.data_manager.get_data(s["id"]).collapse()
                    if not has_xticks:
                        x_ticks_labels = data[0]
                        ax_main.set_xticks(np.arange(len(x_ticks_labels)))
                        #x_ticks_labels = [ l if l not in PARTICLE_MAP else PARTICLE_MAP[l] for l in x_ticks_labels ]
                        ax_main.set_xticklabels(x_ticks_labels, fontsize=self.default_font_size)
                        ax_main.set_xlabel(plot_config["x_axes"].get("label", self.default_x_axis_label), fontsize=self.default_font_size)
                        #ax_main.tick_params('y', colors=colors[i%len(colors)], labelsize=global_font_size)
                        #ax_main.set_ylabel(axis_titles[i], color=colors[i%len(colors)], fontsize=global_font_size)
                        plt.setp(ax_main.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

                        # TODO That is for now brute-force. See above TODO
                        break
            else:
                ax = ax_main.twinx()

            axis = y_axis.get("axis", {})
            y_label = axis.get("label", self.default_y_axis_label)
            style = axis.get("style", {})
            ax.set_ylabel(y_label, color=style.get("color", "b"), fontsize=self.default_font_size)
            ax.tick_params('y', colors=style.get("color", "b"), labelsize=self.default_font_size)
            for s in y_axis["sources"]:
                plot_prop = deepcopy(s.get("custom", {}))
                plot_prop.pop("plot_type")
                data = self.data_manager.get_data(s["id"]).collapse()
                y_data = data[1]

                plot_actions = s.get("actions", {})
                if "scale_to" in plot_actions:
                    scale_to = plot_actions["scale_to"]
                    multiply_by = scale_to / np.sum(y_data)
                    y_data = y_data * multiply_by

                # TODO If labeled axis the order might be different. Account for that
                ax.scatter(np.arange(len(y_data)), y_data,
                           s=dpi * 14, **plot_prop)
        for ext in plot_config.get("save_extensions", ["eps"]):
            fig.savefig(f"{name}.{ext}")

    def plot_2d(self, name, plot_config):
        labels_x = []
        labels_y = []

        data = None

        for y_axis in plot_config["y_axes"]:
            # TODO Make sure sources names appear only once
            for s in y_axis["sources"]:
                data = self.data_manager.get_data(s["id"]).collapse()
                
                # Make unique lists
                x_coords = list(set(data[1]))
                y_coords = list(set(data[0]))

                labels_to_index_x = { x_coords[i]: i for i in range(len(x_coords)) }
                labels_to_index_y = { y_coords[i]: i for i in range(len(y_coords)) }

                heatmap_array = np.zeros(len(x_coords) * len(y_coords)).reshape(len(x_coords), len(y_coords))
                

                for c1, c2, v in zip(data[1], data[0], data[2]):
                    heatmap_array[labels_to_index_x[c1],labels_to_index_y[c2]] = v

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
                #axis = y_axis.get("axis", {})
                #y_label = axis.get("label", self.default_y_axis_label)

                # TODO If labeled axis the order might be different. Account for that
                x_coords = np.array(x_coords)[indices]
                cbar_label = "something"
                # normalize
                #if args.normalize:
                heatmap_array /= heatmap_array.sum()
                
                custom = plot_config.get("custom", {})

                dpi = custom.get("dpi", self.default_dpi)

                fig = plt.figure(tight_layout=True,
                                 figsize=(custom.get("width", self.default_figure_height),
                                          custom.get("height", self.default_figure_height)), dpi=dpi)

                ax = fig.add_subplot()

                im = ax.imshow(np.transpose(heatmap_array), cmap="Reds")

                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=self.default_font_size)
                cbar.ax.tick_params(labelsize=self.default_font_size)


                ax.set_xticks(np.arange(len(x_coords)))
                ax.set_yticks(np.arange(len(y_coords)))

                # Replace with beautiful labels if possible
                #labels_x = [ replace_tex(l) if l not in PARTICLE_MAP else PARTICLE_MAP[l] for l in labels_x ]
                #print(labels_x)

                ax.set_xticklabels(x_coords, fontsize=self.default_font_size)
                ax.set_yticklabels(y_coords, fontsize=self.default_font_size)

                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=self.default_font_size)

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

                ax.set_xticks(np.arange(len(x_coords)+1)-.5, minor=True)
                ax.set_yticks(np.arange(len(y_coords)+1)-.5, minor=True)
                ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
                ax.tick_params(which="minor", bottom=False, left=False)

                fig.tight_layout()
                for ext in plot_config.get("save_extensions", ["eps"]):
                    fig.savefig(f"{name}.{ext}")
                break
            break


    def plot_all(self):
        plots = self.config.get_plots()
        
        for n, p in plots.items():
            self.logger.info("Plot %s", n)
            dim = p.get("dimension", 1)
            if dim == 1:
                self.plot_1d(n, p)
            elif dim == 2:
                self.plot_2d(n, p)







                    















