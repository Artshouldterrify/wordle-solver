from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from heapq import nlargest
from wordle_feedback import matches_feedback_history


def _build_valid_guess_list() -> List[str]:
    guesses = []
    with open("wordle_answers.txt", "r") as file:
        guesses.extend(line.strip() for line in file)
    with open("wordle_guesses.txt", "r") as file:
        guesses.extend(line.strip() for line in file)
    return guesses


class StatisticalSolver:
    """Choose guesses using word frequency weights and Wordle feedback."""

    def __init__(self, word_length: int = 5):
        self.active_words = None
        self.word_length = word_length
        self.valid_word_list = [
            word for word in _build_valid_guess_list()
            if len(word) == word_length
        ]
        self.scaling_factors = self._calculate_scaling_factors()
        self._word_letter_counts = {
            word: {letter: word.count(letter) for letter in set(word)}
            for word in self.valid_word_list
        }
        self.guess_record: List[str] = []
        self._reset_state()

    @staticmethod
    def _transform_scaling_factors(
        scaling_factors: Dict[str, float], a: float = 0, b: float = 1
    ) -> Dict[str, float]:
        sorted_words = sorted(scaling_factors, key=scaling_factors.get)
        x_line = np.linspace(-10, 10, len(sorted_words))
        y_values = 1.0 / (1 + np.exp(-b * x_line + a))
        return dict(zip(sorted_words, y_values))

    def _calculate_scaling_factors(self) -> Dict[str, float]:
        frequency_df = pd.read_csv("unigram_freq.csv")
        frequency_df["word"] = frequency_df["word"].str.lower()
        frequency_df = frequency_df[
            frequency_df["word"].isin(self.valid_word_list)
        ]
        frequencies = dict(zip(frequency_df["word"], frequency_df["count"]))
        # Keep every valid guess selectable when the frequency file is incomplete.
        frequencies.update({
            word: 0 for word in self.valid_word_list if word not in frequencies
        })
        return self._transform_scaling_factors(frequencies)

    def _reset_state(self) -> None:
        self.correct_letters: Dict[int, str] = {}
        self.letters_in_word: Dict[str, int] = {}
        self.letters_not_in_word = set()
        self.letter_pos_not_possible: List[Tuple[str, int]] = []
        self.feedback_history: List[List[str]] = []
        self.active_words = self.valid_word_list.copy()

    def reset(self) -> None:
        self.guess_record = []
        self._reset_state()

    def update_state(self, feedback: List[str]) -> None:
        self.feedback_history.append(feedback.copy())
        self.active_words = [
            word for word in self.active_words
            if matches_feedback_history(word, self.guess_record, self.feedback_history)
        ]

    def next_guess(self) -> str:
        guesses = {
            word: self.scaling_factors.get(word, 0.0)
            for word in self.active_words
        }
        guesses = {word: score for word, score in guesses.items() if score > 0}
        if not guesses:
            raise RuntimeError("No valid guesses remain.")
        top_guesses = nlargest(10, guesses, key=guesses.get)
        probabilities = np.array([guesses[word] for word in top_guesses])
        probabilities /= probabilities.sum()
        guess = str(np.random.choice(top_guesses, p=probabilities))
        self.guess_record.append(guess)
        return guess
