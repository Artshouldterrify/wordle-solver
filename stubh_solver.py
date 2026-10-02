from typing import Any, Optional, List, Dict
import pandas as pd
import numpy as np
import nltk
import time
import os
from wordle_feedback import matches_feedback_history


def _calc_most_common_bigrams() -> pd.DataFrame:
    nltk.download('words')
    from nltk.corpus import words
    five_letter_words = [x for x in words.words() if len(x) == 5]
    bigrams: Dict[str, int] = {}
    for word in five_letter_words:
        for i in range(4):
            bigram = word[i:i+2]
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
    sorted_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_bigrams, columns=["bigram", "count"])

def _store_common_bigrams():
    if not os.path.exists("common_bigrams.csv"):
        df = _calc_most_common_bigrams()
        df.to_csv("common_bigrams.csv", index=False)
    return

def _build_valid_guess_list() -> List[str]:
    guesses = []
    with open('wordle_answers.txt', 'r') as f:
        for line in f.readlines():
            guesses.append(line.strip())
    with open('wordle_guesses.txt', 'r') as f:
        for line in f.readlines():
            guesses.append(line.strip())
    return guesses

def _build_valid_solution_list() -> List[str]:
    valid_solution_list = []
    with open('wordle_answers.txt', 'r') as f:
        for line in f.readlines():
            valid_solution_list.append(line.strip())
    return valid_solution_list

def _sigmoid(z: float, mu: float, std: float) -> float:
    return 1 / (1 + np.exp(-(z - mu) / std))

def _load_statistical_frequencies(word_l: List[str]) -> pd.DataFrame:
    df = pd.read_csv("unigram_freq.csv").set_index("word")
    df = df[df.index.isin(word_l)].copy()
    mean, std = df["count"].mean(), df["count"].std()
    df["frequency"] = df["count"].apply(lambda y: _sigmoid(y, mean, std))
    df["word_has_y"] = df.index.str.contains("y")
    df["vowels_in_word"] = df.index.str.count(r'[aeiou]')
    df["n_bigrams"] = 0
    df["multiple_letters"] = 0
    df["starts_with_vowel"] = df.index.str[0].isin(_VOWELS).astype(int)
    try:
        bigrams = pd.read_csv("common_bigrams.csv").set_index("bigram")
        bigrams = bigrams[~bigrams.index.str.contains(r"a|e|i|o|u")]
        bigrams = bigrams[bigrams["count"] >= 100]
    except FileNotFoundError:
        raise FileNotFoundError()
    for word, row in df.iterrows():
        for bigram, row_b in bigrams.iterrows():
            if bigram in word:
                df.at[word, "n_bigrams"] += 1
        # repeated letter penalty.
        char_map, penalty = {}, False
        for char in word:
            char_map[char] = char_map.get(char, 0) + 1
            if char_map[char] > 1:
                penalty = True
                break
        if penalty:
            df.at[word, "multiple_letters"] = 1
    df["active"] = 1
    return df.sort_values(by="frequency", ascending=False)

_VOWELS = {"a", "e", "i", "o", "u"}

class StubhSolver:
    def __init__(self, use_expanded_list: bool = False):
        self.min_limits, self.max_limits = {}, {}
        self.chars_incorrect_pos: Dict[str, set] = {}
        self.correct_letters: Dict[str, int] = {}
        self.guess_count = 1
        self.guess_record = []
        self.feedback_history: List[List[str]] = []
        self.use_expanded_list = use_expanded_list
        if os.path.exists("common_bigrams.csv"):
            _store_common_bigrams()
        if self.use_expanded_list:
            self.valid_word_list = _build_valid_guess_list()
        else:
            self.valid_word_list = _build_valid_solution_list()
        self.frequency_df = _load_statistical_frequencies(self.valid_word_list)

    def reset(self):
        self.min_limits, self.max_limits = {}, {}
        self.chars_incorrect_pos: Dict[str, set] = {}
        self.correct_letters: Dict[str, int] = {}
        self.guess_record = []
        self.feedback_history = []
        self.guess_count = 1
        if self.use_expanded_list:
            self.valid_word_list = _build_valid_guess_list()
        else:
            self.valid_word_list = _build_valid_solution_list()
        self.frequency_df["active"] = 1
        return

    def _update_valid_word_list(self) -> None:
        updated_valid_word_list = []
        for word in self.valid_word_list:
            inc = True
            # green letters should be 1:1 matched otherwise simple exclusion from candidate list.
            for k, v in self.correct_letters.items():
                if word[k] != v:
                    inc = False
                    self.frequency_df.loc[word, "active"] = 0
                    break
            if not inc:
                continue
            skip_indexes = set(self.correct_letters.keys())
            # collect all non-green indexes in a map.
            word_char_map = {}
            for i in range(len(word)):
                if i in skip_indexes:
                    continue
                if i not in self.correct_letters.keys():
                    word_char_map[word[i]] = word_char_map.get(word[i], 0) + 1
            # check the character map to see if it violates imposed limits.
            for k,v in word_char_map.items():
                if k in self.max_limits.keys():
                    if word_char_map[k] > self.max_limits[k]:
                        inc = False
                        self.frequency_df.loc[word, "active"] = 0
                        break
            for k,v in self.min_limits.items():
                if k not in word_char_map.keys():
                    inc = False
                    self.frequency_df.loc[word, "active"] = 0
                    break
                if word_char_map[k] < v:
                    inc = False
                    self.frequency_df.loc[word, "active"] = 0
                    break
            if not inc:
                continue
            # lastly, check if an existing letter is in a position we know to be incorrect.
            for i in range(len(word)):
                if i in skip_indexes:
                    continue
                if word[i] in self.chars_incorrect_pos.keys():
                    if i in self.chars_incorrect_pos[word[i]]:
                        inc = False
                        self.frequency_df.loc[word, "active"] = 0
                        break
            if not inc:
                continue
            updated_valid_word_list.append(word)
        self.valid_word_list = updated_valid_word_list
        return

    @staticmethod
    def _first_guess():
        return "crate"

    def _second_guess(self):
        ALPHA, BETA, GAMMA = 0.5, 0.5, 0.3
        ALPHA_P, BETA_P = 0.1, 0.1
        active_df = self.frequency_df[self.frequency_df["active"] == 1]
        active_df["score"] = ((ALPHA * active_df["vowels_in_word"] + BETA * active_df["n_bigrams"] + GAMMA * active_df["word_has_y"])
                                / (ALPHA_P * active_df["starts_with_vowel"] + BETA_P * active_df["multiple_letters"] + 1)) * active_df["frequency"]
        active_df.sort_values(by="score", ascending=False, inplace=True)
        if len(active_df) >= 10:
            records = active_df.iloc[:10]
        else:
            records = active_df
        probs, words = records["score"].values, records.index.values
        total_probability = np.sum(probs)
        if not len(words):
            raise RuntimeError("No valid guesses remain.")
        if total_probability <= 0:
            return str(words[0])
        probs = probs / total_probability
        return np.random.choice(words, p=probs)

    def _guess_3_correct(self):
        active_df = self.frequency_df[self.frequency_df["active"] == 1]
        if len(active_df) >= 10:
            records = active_df.iloc[:10]
        else:
            records = active_df
        probs, words = records["frequency"].values, records.index.values
        total_probability = np.sum(probs)
        if not len(words):
            raise RuntimeError("No valid guesses remain.")
        if total_probability <= 0:
            return str(words[0])
        probs = probs / total_probability
        return np.random.choice(words, p=probs)

    def next_guess(self) -> str:
        if self.guess_count == 1:
            guess = self._first_guess()
        elif len(self.correct_letters) >= 3:
            guess = self._guess_3_correct()
        else:
            guess = self._second_guess()
        self.guess_count += 1
        self.guess_record.append(guess)
        return guess

    def update_state(self, feedback: List[str]):
        self.feedback_history.append(feedback.copy())
        self.correct_letters = {
            position: guess[position]
            for guess, pattern in zip(self.guess_record, self.feedback_history)
            for position, result in enumerate(pattern)
            if result == "G"
        }
        self.min_limits = {}
        self.max_limits = {}
        self.chars_incorrect_pos = {}
        for guess, pattern in zip(self.guess_record, self.feedback_history):
            present_counts = {}
            for letter, result in zip(guess, pattern):
                if result in {"G", "Y"}:
                    present_counts[letter] = present_counts.get(letter, 0) + 1
            for letter, count in present_counts.items():
                self.min_limits[letter] = max(self.min_limits.get(letter, 0), count)
            for position, (letter, result) in enumerate(zip(guess, pattern)):
                if result == "Y" or (result == "X" and letter in present_counts):
                    self.chars_incorrect_pos.setdefault(letter, set()).add(position)
                if result == "X":
                    maximum = present_counts.get(letter, 0)
                    self.max_limits[letter] = min(self.max_limits.get(letter, maximum), maximum)
        self.valid_word_list = [
            word for word in self.valid_word_list
            if matches_feedback_history(word, self.guess_record, self.feedback_history)
        ]
        self.frequency_df["active"] = self.frequency_df.index.isin(self.valid_word_list).astype(int)
        return


if __name__ == "__main__":
    pass