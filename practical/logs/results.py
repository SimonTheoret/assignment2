#!/usr/bin/env python3
import ast
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns


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
        self.exps: dict[str, dict[str, Any]] = {}

    def find_all_exps(self):
        """Create a dict out of the experiments and store that dict in self.exps ."""
        for dir in self.dir_names:
            new_dir = {}
            for file in self.result_files_names:
                f = open(f"{self.init_dir}/{dir}/{file}", "r")
                if file == "avg_mem_percentage_used.txt" or file == "avg_mem_used.txt":
                    new_dir[file] = ast.literal_eval(f.read())
                else:
                    content = list(filter(None, f.read().split("\n")))
                    new_dir[file] = [float(x) for x in content]
            self.exps[dir] = new_dir


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
        files_x: list[str],
        dirs_y: list[str],
        files_y: list[str],
        main_title: str,
        titles: list[str],
        legends: list[str],
        xlabels: list[str],
        ylabels: list[str],
    ):
        sns.set()
        fig, axes = plt.subplots(*dim)

        for i in range(dim[0] * dim[1]):
            ax = axes[i]
            x = self.data[dirs_x[i]][files_x[i]]
            y = self.data[dirs_y[i]][files_y[i]]
            ax.plot(x, y)
            ax.set_title(titles[i])
            ax.legend(legends[i])
            ax.set_ylabel(ylabels[i])
            ax.set_xlabel(xlabels)

        fig.suptitle(main_title)
        plt.savefig(self.fig_path / main_title)
        plt.show()

    def _create_dir(self, init_dir: str) -> Path:
        path = Path(init_dir) / "figures/"
        path.mkdir(parents=False, exist_ok=True)
        return path

    def _plot_for_all_exp(
        self,
        dim: tuple[int, int],
        dirs_x: list[str],
        files_x: list[str],
        dirs_y: list[str],
        files_y: list[str],
        main_titles: list[str],
        titles: list[str],
        legends: list[str],
        xlabels: list[str],
        ylabels: list[str],
    ):
        for i,dir in enumerate(self.data_dir.dir_names):
            dir_list_x = [dir]*(dim[0]*dim[1])
            dir_list_y = [dir]*(dim[0]*dim[1])
            self.plot(
                dim,
                dir_list_x,
                files_x,
                dir_list_y,
                files_y,
                main_titles[i], # 1 to 1 relation with data_dir.dir_names
                titles,
                legends,
                xlabels,
                ylabels,
            )
        pass


if __name__ == "__main__":
    dd = DataDir("/home/simon/Documents/a2023/DL/assignment2/practical/logs")
    dd.find_all_exps()
    # for _,ele in dd.exps.items():
    #     for _, el in ele.items():
    #         print(type(el[0]))
    plotter = Plotter(dd)
    plotter.plot(
        (2, 1),
        ["gpt1_layer_1_adam", "gpt1_layer_1_adam"],
        ["valid_loss.txt", "valid_loss.txt"],
        ["gpt1_layer_1_adam", "gpt1_layer_1_adam"],
        ["train_loss.txt", "train_loss.txt"],
        "gros test",
        ["test", "test2"],
        ["test", "textf2"],
        ["x", "encore x"],
        ["y", "encore y"],
    )
