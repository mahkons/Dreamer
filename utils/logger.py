import pandas as pd
import torch
import os
import numpy as np
from collections import defaultdict
import time
import shutil
import datetime

logger__ = None

def init_logger(logdir, logname):
    global logger__
    logger__ = Logger(logdir, logname)

def log():
    global logger__
    return logger__

class Logger():
    def __init__(self, logdir, logname):
        self.logdir = logdir

        if logname.startswith("tmp") and os.path.exists(os.path.join(logdir, logname)):
            shutil.rmtree(os.path.join(logdir, logname))
        
        assert(os.path.isdir(logdir))
        self.dir = os.path.join(logdir, logname)
        os.mkdir(self.dir)

        self.plots = dict()
        self.plots_columns = dict()
        self.max_rows = 10**4
        self.cur_rows = 0

        self.time_metrics = defaultdict(float)
        self.prev_time = None

    def get_log_path(self):
        return self.dir

    def add_plot(self, name, columns):
        assert name not in self.plots
        self.plots[name] = list()
        self.plots_columns[name] = list(columns) + ["time"]

    def add_plot_point(self, name, point):
        cur_time_str = str(datetime.datetime.now())
        point = list(point) + [cur_time_str]
        
        self.plots[name].append(point)
        self.cur_rows += 1
        if self.cur_rows >= self.max_rows:
            self.save_logs()

    def add_plot_points(self, name, points):
        cur_time_str = str(datetime.datetime.now())
        points = [list(point) + [cur_time_str] for point in points]

        self.plots[name].extend(points)
        self.cur_rows += len(points)
        if self.cur_rows >= self.max_rows:
            self.save_logs()



    def check_time(self, add_to_value=None):
        now = time.time()
        if add_to_value:
            self.time_metrics[add_to_value] += now - self.prev_time
        self.prev_time = now
        return now

    def zero_time_metrics(self):
        self.time_metrics.clear()

    def print_time_metrics(self):
        print("=" * 50)
        print("=" * 18 + " Time metrics " + "=" * 18)
        for k, v in self.time_metrics.items():
            print("{}: {} seconds".format(k, v))
        print("=" * 50)

    def save_csv(self):
        plot_path = os.path.join(self.dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        for plot_name, plot_data in self.plots.items():
            filename = os.path.join(plot_path, plot_name + ".csv")
            pd.DataFrame(plot_data, columns=self.plots_columns[plot_name]).to_csv(filename, index=False, mode='a', \
                    header=not os.path.exists(filename)) # append


    # clears saved logs
    def save_logs(self):
        self.save_csv()
        self.clear_logs()

    def clear_logs(self):
        self.cur_rows = 0
        for key, value in self.plots.items():
            value.clear()
        self.time_metrics.clear()
