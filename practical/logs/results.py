#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
import ast


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


if __name__ == "__main__":
    dd = DataDir("/home/simon/Documents/a2023/DL/assignment2/practical/logs")
    dd.find_all_exps()
    print(dd.exps["gpt1_layer_1_adam"]["train_time.txt"])
    print(type(dd.exps["gpt1_layer_1_adam"]["avg_mem_used.txt"]))
    print(dd.exps)
