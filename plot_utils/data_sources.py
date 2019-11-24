import numpy as np
import pandas as pd
from ROOT import TH1, TH2, TFile
from plot_utils.io import print_dict
from plot_utils.logger import get_logger

class DataSource:
    def __init__(self, dimensions):
        
        self.logger = get_logger()

        self.coord_columns = range(dimensions + 1)
        # TODO Add errors
        #self.error_columns = range(dimensions + 1, 3 * (dimensions + 1))
        self.dimensions = dimensions

        self.df = pd.DataFrame(columns=self.coord_columns)

    def add(self, coords, value):
        if len(coords) != self.dimensions:
            self.logger.fatal("Dimension mismatch of given point with dimension %i and data source with dimension %i",
                              len(coords), self.dimensions)
        add_dict = {col: v for col, v in zip(self.coord_columns, coords)}
        add_dict[self.dimensions] = value
        # TODO This is super inefficient as it makes a copy of the entire df
        self.df = self.df.append(add_dict, ignore_index=True)

    def extend(self, to_be_extended):
        ext_dict = {}
        for i in range(self.dimensions):
            ext_dict[i] = [ext[0][i] for ext in to_be_extended]
        ext_dict[self.dimensions] = [ext[1] for ext in to_be_extended]
        df_ext = pd.DataFrame(ext_dict)
        print(self.df)
        print(df_ext)
        # TODO This is super inefficient as it makes a copy of the entire df
        self.df = pd.concat([self.df, df_ext], ignore_index=True)
        print(self.df)

    #def extend(self, to_be_added):


    def __getitem__(self, key):
        return self.df[key].values

    def collapse(self):
        """
        This is basically binning the dataframe and returning a new one DataSource with the binned one
        For numeric columns 10 bins is the default. For labeled coordinates it's obvious but maybe customised as well.
        """
        if self.df.empty:
            self.logger.warning("Nothing to collaps")
            return self
        # Limits are None for labeled corrdinates. So let's fill a list with (low, up) tuples
        limits = []
        
        n_bins = 10

        def map_number_to_bin(num, edges):
            for i in range(len(edges), -1, -1):
                if num > edges[i]:
                    return i
        def map_bin_to_center(ibin, centers):
            if centers is None:
                return ibin
            return centers[ibin]

        vec_number_to_bin = np.vectorize(map_number_to_bin, excluded=["edges"])

        new_coord_columns = []
        bin_edges = []

        for c in range(self.dimensions):
            try:
                # TODO Only relying on the first entry in a column. Should we fix the data type?
                number = float(self.df[c][0])
                limit_up = max(self.df[c])
                limit_low = min(self.df[c])
                width = (limit_up - limit_low) / n_bins
                edges = [limit_low + i * width for i in range(n_bins)]
                new_coord_columns.append(vec_number_to_bin(self.df[c].values, edges))
                bin_edges.append(edges + [limit_up])
            except ValueError:
                new_coord_columns.append(self.df[c].values)
                bin_edges.append(None)

        new_coord_columns.append(self.df[self.dimensions])
        
        # This can be done smarter with a generator such that we don't have to save all columns as an intermediate step
        df_bins = pd.DataFrame({c: v for c, v in zip(self.coord_columns, new_coord_columns)})
        df_bins = df_bins.groupby(list(range(self.dimensions)), as_index=False).agg({self.dimensions: ["sum"]})

        
        vec_bin_to_center = np.vectorize(map_bin_to_center, excluded=["centers"])
        for c in range(self.dimensions):
            centers = [(bin_edges[c][i+1] - bin_edges[c][i]) / 2. for i in range(len(bin_edges[c]) - 1)] if bin_edges[c] is not None else None
            df_bins[c] = vec_bin_to_center(df_bins[c].values, centers)


        new_data = DataSource(self.dimensions)
        new_data.df = df_bins

        return new_data


def digest_root_source(**kwargs):
    # NOTE TODO Can for now only extract TH1 and TH2

    logger = get_logger()

    def digest_1d(histo):
        to_be_added = []
        axis = histo.GetXaxis()

        get_coord = None

        if axis.IsAlphanumeric():
            get_coord = axis.GetBinLabel
        else:
            get_coord = axis.getBinCenter

        for ibin in range(1, axis.GetNbins() + 1):
            coord = get_coord(ibin)
            if coord == "":
                continue
            to_be_added.append(((coord,), histo.GetBinContent(ibin)))

        return to_be_added

    def digest_2d(histo):
        to_be_added = []
        x_axis = histo.GetXaxis()
        y_axis = histo.GetYaxis()

        get_coord_x = None
        get_coord_y = None

        if x_axis.IsAlphanumeric():
            get_coord_x = x_axis.GetBinLabel
        else:
            get_coord_x = x_axis.getBinCenter
        if y_axis.IsAlphanumeric():
            get_coord_y = y_axis.GetBinLabel
        else:
            get_coord_y = y_axis.getBinCenter

        for ibin in range(1, x_axis.GetNbins() + 1):
            for jbin in range(1, y_axis.GetNbins() + 1):
                coord_x = get_coord_x(ibin)
                coord_y = get_coord_y(ibin)
                if coord_x == "" or coord_y == "":
                    continue
                to_be_added.append(((coord_x, coord_y,), histo.GetBinContent(ibin, jbin)))

        return to_be_added


    files = kwargs.pop("files", None)
    in_root_dir = kwargs.pop("dir", None)
    name = kwargs.pop("name", None)

    if None in [files, in_root_dir, name]:
        logger.fatal("Not all fields provided")

    if kwargs:
        logger.warning("There are unknown arguments when digesting ROOT file which will be ignored")
        print_dict(kwargs)

    # TODO For now treating everything as 1d histogram
    data = None
    for f in files:
        root_file = TFile.Open(f, "READ")
        root_dir = root_file.Get(in_root_dir)
        histo = root_dir.Get(name)
        if isinstance(histo, TH2):
            if data is None:
                data = DataSource(2)
            to_be_added = digest_2d(histo)
        else:
            if data is None:
                data = DataSource(1)
            to_be_added = digest_1d(histo)
        data.extend(to_be_added)

    return data



DIGEST_METHODS = {"ROOT_File": digest_root_source}







