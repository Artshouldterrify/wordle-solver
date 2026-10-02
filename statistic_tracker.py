import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

_PATH = "saved_data/"

class StatisticTracker:
    def __init__(self, save_path: str = _PATH, file_name: str = "statistic_tracking.csv"):
        self.save_path = save_path
        self.file_name = file_name
        self.tracking_df = pd.DataFrame(columns=["solution_word","guess_list","is_solved","n_guesses"])

    def add_record(self, was_solved: bool, solution_word: str, n_guesses: int, guess_list: Optional[List[str]] = None):
        if guess_list is None:
            guess_list = []
        new_record = pd.DataFrame([{
            "solution_word": solution_word,
            "guess_list": guess_list,
            "is_solved": was_solved,
            "n_guesses": n_guesses
        }])
        self.tracking_df = pd.concat([self.tracking_df, new_record], ignore_index=True)
        return

    def reset(self):
        self.tracking_df = pd.DataFrame(columns=["solution_word","guess_list","is_solved","n_guesses"])
        return

    def save(self, file_name: str = None):
        if file_name is None:
            file_name = self.file_name
        self.tracking_df.to_csv(self.save_path + file_name, index=False)
        return

    def set_file_name(self, file_name: str):
        self.file_name = file_name
        return

    def set_save_path(self, save_path: str):
        self.save_path = save_path
        return

    def summary(self):
        total_games = len(self.tracking_df)
        solved_games = self.tracking_df["is_solved"].sum()
        unsolved_games = total_games - solved_games
        percentage_solved = solved_games / total_games * 100 if total_games else 0
        avg_guesses = self.tracking_df[self.tracking_df["is_solved"]]["n_guesses"].mean() if solved_games > 0 else 0
        summary_dict = {
            "total_games": total_games,
            "solved_games": solved_games,
            "unsolved_games": unsolved_games,
            "avg_guesses": avg_guesses,
            "percentage_solved": percentage_solved
        }
        self.save()
        return summary_dict