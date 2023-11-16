#!/usr/bin/env python3
import ast
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import copy


class DataDir:
    def __init__(self, current_dir):
        self.init_dir = current_dir
        self.dir_names: list = [
            "gpt1_layer_1_adam",
            "gpt1_layer_1_adamw",
            "gpt1_layer_1_momentum",
            "gpt1_layer_1_sgd",
            "gpt1_layer_2_adamw",
            "gpt1_layer_4_adamw",
            "lstm_layer_1_adam",
            "lstm_layer_1_adamw",
            "lstm_layer_1_momentum",
            "lstm_layer_1_sgd",
            "lstm_layer_2_adamw",
            "lstm_layer_4_adamw",
        ]
        self.result_files_names: list = [
            "avg_mem_percentage_used.txt",
            "avg_mem_used.txt",
            "test_loss.txt",
            "test_ppl.txt",
            "test_time.txt",
            "train_loss.txt",
            "train_ppl.txt",
            "train_time.txt",
            "valid_loss.txt",
            "valid_ppl.txt",
            "valid_time.txt",
        ]

        self.files: list = [
            "avg_mem_percentage_used.txt",
            "avg_mem_used.txt",
            "test_loss.txt",
            "test_ppl.txt",
            "test_time.txt",
            "train_loss.txt",
            "train_ppl.txt",
            "train_time.txt",
            "valid_loss.txt",
            "valid_ppl.txt",
            "valid_time.txt",
        ]

        self.exps: dict[str, dict[str, Any]] = {}

    def find_all_exps(self):
        """Create a dict out of the experiments and store that dict in self.exps ."""
        for dir in self.dir_names:
            new_dir = {}
            for file in self.result_files_names:
                f = open(f"{self.init_dir}/{dir}/{file}", "r")
                if file == "avg_mem_percentage_used.txt" or file == "avg_mem_used.txt":
                    new_dir[file] = ast.literal_eval(f.read())
                elif file == "test_time.txt" or file == "train_time.txt" or file == "valid_time.txt":
                    content = list(filter(None, f.read().split("\n")))
                    new_dir[file] = [float(x) for x in content]
                    a = list(np.array(new_dir[file]).cumsum())
                    name = "cumulative_"+file
                    self.files.append(name)
                    new_dir[name] = a
                else:
                    content = list(filter(None, f.read().split("\n")))
                    new_dir[file] = [float(x) for x in content]
            self.exps[dir] = new_dir

    def print_avg_times(self):
        for dir in self.dir_names:
            print("-----------------")
            print(dir)
            exp = self.exps[dir]
            train_time = exp["train_time.txt"]
            avg_train_time = sum(train_time)/len(train_time)
            print(f"average train time: { avg_train_time }")

            valid_time = exp["valid_time.txt"]
            avg_valid_time = sum(valid_time)/len(valid_time)
            print(f"average validation time: { avg_valid_time }")

            test_time = exp["test_time.txt"]
            avg_test_time = sum(test_time)/len(test_time)
            print(f"average test time: { avg_test_time }")

            train_loss = exp["train_loss.txt"][-1]
            valid_loss = exp["valid_loss.txt"][-1]
            test_loss = exp["test_loss.txt"][-1]
            print(f"train loss: {train_loss}, validation loss: {valid_loss}, test loss: {test_loss}")

            train_ppl = exp["train_ppl.txt"][-1]
            valid_ppl = exp["valid_ppl.txt"][-1]
            test_ppl = exp["test_ppl.txt"][-1]
            print(f"train ppl: {train_ppl}, validation ppl: {valid_ppl}, test ppl: {test_ppl}")

    def print_valid_ppl(self):
        for dir in self.dir_names:
            print("-----------------")
            print(dir)
            exp = self.exps[dir]
            min_valid = np.min( np.array(exp['valid_ppl.txt']))
            print(f"min validation perplexity: {min_valid}")

    def print_all_gpu_mem(self):
        for dir in self.dir_names:
            print("-----------------")
            print(dir)
            exp = self.exps[dir]
            avg_mem = exp["avg_mem_used.txt"]
            avg_mem = sum(avg_mem[1:])/len(avg_mem[1:])
            avg_per_mem = exp["avg_mem_percentage_used.txt"]
            avg_per_mem = sum(avg_per_mem[1:])/len(avg_per_mem[1:])
            print(f"average memory used: {avg_mem}")
            print(f"average memory used: {avg_per_mem}")


class Plotter:
    def __init__(self, dd: DataDir):
        if dd.exps == None:
            dd.find_all_exps()
        self.data = dd.exps
        self.data_dir = dd
        self.fig_path: Path = self._create_dir(dd.init_dir)

    def plot(
        self,
        dim: tuple[int, int],
        dirs_x: list[str],
        files_x: list,
        dirs_y: list,
        files_y: list[str],
        main_title: str,
        titles: list[str],
        legends: list[str],
        xlabels: list[str],
        ylabels: list[str],
        show: bool = True
    ):
        sns.set()
        fig, axes = plt.subplots(*dim, figsize = (12,15))

        for i in range(0,dim[0]):
            for j in range(0,dim[1]):
                ax = axes[i,j]
                k  = i + j
                if i == 1:
                    k+=1
                if isinstance(files_x[k], str):
                    x = self.data[dirs_x[k]][files_x[k]]
                else:
                    x = files_x[k]
                if isinstance(files_y[k], str):
                    y = self.data[dirs_y[k]][files_y[k]]
                else:
                    y = files_y[k]
                ax.plot(x, y, label = legends[k])
                ax.set_title(titles[k])
                ax.legend()
                ax.set_ylabel(ylabels[k])
                ax.set_xlabel(xlabels[k])

        fig.suptitle(main_title)
        plt.savefig(self.fig_path / main_title.replace(" ", "_"))
        if show:
            plt.show()

    def _create_dir(self, init_dir: str) -> Path:
        path = Path(init_dir) / "figures/"
        path.mkdir(parents=False, exist_ok=True)
        return path

    def plot_for_all_exp(
        self,
        dim: tuple[int, int],
        # dirs_x: list[str],
        files_x: list,
        # dirs_y: list[str],
        files_y: list,
        main_titles: list[str],
        titles: list[str],
        legends: list[str],
        xlabels: list[str],
        ylabels: list[str],
        show: bool = True
    ):
        for i, dir in enumerate(self.data_dir.dir_names):
            dir_list_x = [dir] * (dim[0] * dim[1])
            dir_list_y = [dir] * (dim[0] * dim[1])
            self.plot(
                dim,
                dir_list_x,
                files_x,
                dir_list_y,
                files_y,
                main_titles[i],  # 1 to 1 relation with data_dir.dir_names
                titles,
                legends,
                xlabels,
                ylabels,
                show = False
            )


if __name__ == "__main__":
    dd = DataDir("/home/simon/Documents/a2023/DL/assignment2/practical/logs")
    dd.find_all_exps()
    plotter = Plotter(dd)
    dim = (2, 2)
    files_x = [
        "cumulative_train_time.txt",
        "cumulative_train_time.txt",
        list(range(0, 10)),
        list(range(0, 10)),
    ]
    files_y = ["train_ppl.txt", "valid_ppl.txt", "train_ppl.txt", "valid_ppl.txt"]
    titles = [
        "Training perplexity over time",
        "Validation perplexity over time",
        "Training perplexity over epochs",
        "Validation perplexity over epochs",
    ]
    legends = ["training", "validation", "training", "validation"]
    main_titles = [x.replace("_", " ") for x in plotter.data_dir.dir_names]
    xlabels = ["time (secs)", "time (secs)", "epoch", "epoch"]
    ylabels = ["perplexity", "perplexity", "perplexity", "perplexity"]
    # plotter.plot_for_all_exp(dim, files_x, files_y, main_titles, titles, legends, xlabels, ylabels, show = False)

    # dd.print_avg_times()
    # dd.print_valid_ppl()
    dd.print_all_gpu_mem()
