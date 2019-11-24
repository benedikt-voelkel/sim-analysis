"""
main script for doing data processing, machine learning and analysis
"""
import multiprocessing as mp
from os import makedirs
from os.path import join, exists, expanduser
from plot_utils.logger import get_logger

def make_dirs(dir_to_make):
    """
    Helper function to checking and making a directory
    """
    dir_to_make = expanduser(dir_to_make)
    if not exists(dir_to_make):
        get_logger().info("Make directory %s", dir_to_make)
        makedirs(dir_to_make)

def make_latex_table(column_names, row_names, rows, caption=None, save_path="./table.tex"):
    caption = caption if caption is not None else "Caption"
    with open(save_path, "w") as f:
        f.write("\\documentclass{article}\n\n")
        f.write("\\usepackage[margin=0.7in]{geometry}\n")
        f.write("\\usepackage[parfill]{parskip}\n")
        f.write("\\usepackage{rotating}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{sidewaystable}\n")
        f.write("\\centering\n")
        # As many columns as we need
        columns = "|".join(["c"] * (len(column_names) + 1))
        f.write("\\begin{tabular}{" + columns + "}\n")
        f.write("\\hline\n")
        columns = "&".join([""] + column_names)
        columns = columns.replace("_", "\\_")
        f.write(columns + "\\\\\n")
        f.write("\\hline\\hline\n")
        for rn, row in zip(row_names, rows):
            row_string = "&".join([rn] + row)
            row_string = row_string.replace("_", "\\_")
            f.write(row_string + "\\\\\n")
        f.write("\\end{tabular}\n")
        caption = caption.replace("_", "\\_")
        f.write("\\caption{" + caption + "}\n")
        f.write("\\end{sidewaystable}\n")
        f.write("\\end{document}\n")

def parallelizer(function, argument_list, maxperchunk, max_n_procs=2):
    """
    A centralized version for quickly parallelizing a function.
    """
    get_logger().info("Parallelizing function %s", function.__name__)
    chunks = [argument_list[x:x+maxperchunk] \
              for x in range(0, len(argument_list), maxperchunk)]
    for chunk in chunks:
        get_logger().info("Processing new chunck size = %i", maxperchunk)
        pool = mp.Pool(max_n_procs)
        _ = [pool.apply_async(function, args=chunk[i]) for i in range(len(chunk))]
        pool.close()
        pool.join()

def get_timestamp_string():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
