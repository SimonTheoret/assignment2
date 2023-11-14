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


class Plotter:
    def __init__(self, dd: DataDir):
        if dd.exps == None:
            dd.find_all_exps()
        self.data = dd.exps
        print(dd.exps)
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
    ):
        sns.set()
        fig, axes = plt.subplots(*dim, figsize = (12,15))

        for i in range(dim[0]):
            for j in range(dim[1]):
                ax = axes[i,j]
                if isinstance(files_x[i+j], str):
                    x = self.data[dirs_x[i+j]][files_x[i+j]]
                else:
                    x = files_x[i+j]
                if isinstance(files_y[i+j], str):
                    y = self.data[dirs_y[i+j]][files_y[i+j]]
                else:
                    y = files_y[i+j]
                ax.plot(x, y, label = legends[i+j])
                ax.set_title(titles[i+j])
                ax.legend()
                ax.set_ylabel(ylabels[i+j])
                ax.set_xlabel(xlabels[i+j])

        fig.suptitle(main_title)
        plt.savefig(self.fig_path / main_title)
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
            )


if __name__ == "__main__":
    dd = DataDir("/home/simon/Documents/a2023/DL/assignment2/practical/logs")
    dd.find_all_exps()
    # for _,ele in dd.exps.items():
    #     for _, el in ele.items():
    #         print(type(el[0]))
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
    xlabels = ["time(s)", "time(s)", "epoch", "epoch"]
    ylabels = ["perplexity", "perplexity", "perplexity", "perplexity"]
    plotter.plot_for_all_exp(dim, files_x, files_y, main_titles, titles, legends, xlabels, ylabels)
