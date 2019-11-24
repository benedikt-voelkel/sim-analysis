from copy import deepcopy

from plot_utils.logger import get_logger
from plot_utils.io import parse_yaml

class PlotConfig:
    def __init__(self, config: dict):

        self.logger = get_logger()

        # Static config variables
        self.delim = "::"
        self.this_flag = "THIS"
        self.var_flag = "VAR"
        self.sources_flag = "SOURCES"

        # The raw config dictionary
        self.config_raw = parse_yaml(config)

        # The digested configuration
        self.variables = None
        self.sources = None
        self.plots = None

        self.check_config()
        self.digest()


    def check_config(self):
        if "sources" not in self.config_raw:
            self.logger.fatal("Cannot find \"sources\" field in configuration")
        if "plots" not in self.config_raw:
            self.logger.fatal("Cannot find \"plots\" field in configuration")
        sources_ids = []
        for s in self.config_raw["sources"]:
            if s["id"] in sources_ids:
                self.logger.fatal("There is already a source with id %s", str(s["id"]))
            sources_ids.append(s["id"])


    def make_full_flag(self, *flags):
        return self.delim.join(flags) + self.delim


    def resolve_var(self, name):
        """
        Resolve a potential variable
        """
        var_flag = self.make_full_flag(self.this_flag, self.var_flag)
        if name.find(var_flag) < 0:
            # If not in name, don't do anything
            return name
        if name.find(var_flag) > 0:
            # Has to begin with the variable flag, otherwise invalid
            self.logger.fatal("Invalid variable %s", name)
        if self.variables is None:
            self.logger.fatal("Found variable %s but no variables defined", name)

        var = name.lstrip(var_flag)
        if var not in self.variables:
            self.logger.fatal("Variable %s not defined", name)

        return self.variables[var]


    def digest(self):
        """
        Extract variables, sources and plots from the raw config
        """

        # First extract user defined variables
        if "variables" in self.config_raw:
            self.variables = deepcopy(self.config_raw["variables"])
            # TODO Check that this is a flat dictionary
        self.sources = deepcopy(self.config_raw["sources"])
        self.plots = deepcopy(self.config_raw["plots"])

        for s in self.sources:
            s["type"] = self.resolve_var(s["type"])
            for a in s["args"]:
                # TODO Need to be able to replace not only strings but also strings in lists and dictionaries
                if isinstance(s["args"][a], str):
                    s["args"][a] = self.resolve_var(s["args"][a])
        # TODO Assume no variables used in YAML for plots

    def get_sources(self):
        return self.sources

    def get_plots(self):
        return self.plots
