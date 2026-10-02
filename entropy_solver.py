import time
import numpy as np
import pandas as pd
from typing import Any, Optional, List, Dict, Tuple
from tqdm import tqdm
from wordle_feedback import calculate_wordle_feedback, matches_feedback_history

_FIRST_GUESS_CACHE: Dict[Tuple[bool, bool, float], str] = {}


# utility functions for encoding and decoding from base 3.
def _encode_pattern_to_base_3(pattern: List[str]) -> int:
    encoding, index = 0, 0
    for pos in pattern:
        if pos[0] == 'G':
            encoding += 2 * (3 ** index)
        elif pos[0] == 'Y':
            encoding += 1 * (3 ** index)
        index += 1
    return encoding

def _decode_base_3_to_pattern(encoding_int: int) -> List[str]:
    pattern = []
    for _ in range(5):
        remainder = encoding_int % 3
        if remainder == 2:
            pattern.append('G')
        elif remainder == 1:
            pattern.append('Y')
        else:
            pattern.append('X')
        encoding_int //= 3
    return pattern

# guess list.
def _build_valid_guess_list() -> List[str]:
    guesses = []
    with open('wordle_answers.txt', 'r') as f:
        for line in f.readlines():
            guesses.append(line.strip())
    with open('wordle_guesses.txt', 'r') as f:
        for line in f.readlines():
            guesses.append(line.strip())
    return guesses

def _sigmoid(z: float, mu: float, std: float) -> float:
    return 1 / (1 + np.exp(-(z - mu) / std))

def _load_statistical_frequencies() -> pd.DataFrame:
    df = pd.read_csv("unigram_freq.csv", header=0, index_col=0)
    mean, std = df["count"].mean(), df["count"].std()
    df["frequency"] = df["count"].apply(lambda y: _sigmoid(y, mean, std))
    return df

def _build_valid_solution_list() -> List[str]:
    valid_solution_list = []
    with open('wordle_answers.txt', 'r') as f:
        for line in f.readlines():
            valid_solution_list.append(line.strip())
    return valid_solution_list

def _calculate_pattern(word_1: str, word_2: str) -> List[str]:
    return calculate_wordle_feedback(word_1, word_2)

# function to generate the guess_matrix.
def generate_guess_matrix() -> None:
    valid_guess_list = _build_valid_guess_list()
    sorted_guess_list = sorted(set(valid_guess_list))
    valid_solution_list = set(_build_valid_solution_list())
    matrix = np.zeros([len(sorted_guess_list), len(sorted_guess_list)])
    solution_matrix = np.zeros([len(sorted_guess_list), len(valid_solution_list)])
    progress_bar = tqdm(desc="Comparisons:", total=len(sorted_guess_list)**2)
    for i, word_1 in enumerate(sorted_guess_list):
        k = 0
        for j, word_2 in enumerate(sorted_guess_list):
            pattern = _calculate_pattern(word_1, word_2)
            encoding = _encode_pattern_to_base_3(pattern)
            matrix[i, j] = encoding
            if word_2 in valid_solution_list:
                solution_matrix[i, k] = encoding
                k += 1
            progress_bar.update(1)
    np.savez_compressed("guess_matrix.npz", matrix=matrix, reduced_matrix=solution_matrix)
    return

# pre-compute and store the top 10 guesses for the first guess.
def _pre_compute_entropy_solver_first_guess():
    solver_inst = EntropySolver()
    first_10_guesses = solver_inst.top_10_guesses()
    df = pd.DataFrame(first_10_guesses, columns=["guess", "entropy"])
    df.to_csv("first_10_guesses.csv", index=False)
    return

# Entropy solver class.
class EntropySolver:
    def __init__(self, limit_answer: bool = False, use_frequencies: bool = False, relative_frequency_weight: float = 0.0) -> None:
        if relative_frequency_weight < 0:
            raise ValueError("relative_frequency_weight must be non-negative.")
        self.all_valid_word_list = _build_valid_guess_list()
        self.matrix_word_list = sorted(set(self.all_valid_word_list))
        self.valid_word_list = self.all_valid_word_list.copy()
        self.valid_solution_list = sorted(set(_build_valid_solution_list()))
        self.limit_answer = limit_answer
        self.use_frequencies = use_frequencies
        self.relative_frequency_weight = relative_frequency_weight
        self.frequency_df = _load_statistical_frequencies()
        valid_frequencies = self.frequency_df.reindex(self.all_valid_word_list)["count"]
        self.max_frequency = float(valid_frequencies.max()) if not valid_frequencies.empty else 0.0
        if self.limit_answer:
            self.valid_indexes = np.ones(len(self.valid_solution_list))
        else:
            self.valid_indexes = np.ones(len(self.valid_word_list))
        self.min_limits, self.max_limits = {}, {}
        self.chars_incorrect_pos: Dict[str, set] = {}
        self.correct_letters: Dict[str, int] = {}
        self.feedback_history: List[List[str]] = []
        self.guess_matrix = self._load_guess_matrix(limit_answer)
        self.reduced_index_mapping = {}
        self.word_index_mapping = self._build_word_index_mapping()
        self.guess_record = []
        self.guess_count = 1

    @staticmethod
    def _load_guess_matrix(limit: bool = False) -> np.ndarray:
        try:
            data = np.load("guess_matrix.npz")
        except Exception as e:
            raise FileNotFoundError("guess_matrix.npz not found") from e
        if limit:
            return data["reduced_matrix"]
        return data["matrix"]

    def _build_word_index_mapping(self) -> Dict[str, int]:
        sorted_guess_list = self.matrix_word_list
        valid_solution_list = set(_build_valid_solution_list())
        solution_idx = 0
        mapping = {}
        for i, word in enumerate(sorted_guess_list):
            mapping[word] = i
            if word in valid_solution_list:
                self.reduced_index_mapping[i] = solution_idx
                solution_idx += 1
        return mapping

    def reset(self):
        self.valid_word_list = self.all_valid_word_list.copy()
        self.chars_incorrect_pos: Dict[str, set] = {}
        self.correct_letters: Dict[str, int] = {}
        if self.limit_answer:
            self.valid_indexes = np.ones(len(self.valid_solution_list))
        else:
            self.valid_indexes = np.ones(len(self.valid_word_list))
        self.guess_count = 1
        self.guess_record = []
        self.feedback_history = []
        self.min_limits, self.max_limits = {}, {}
        return

    def _update_valid_word_list(self) -> None:
        updated_valid_word_list = []
        original_length = len(self.valid_word_list)
        for word in self.valid_word_list:
            word_index, inc = self.word_index_mapping[word], True
            # green letters should be 1:1 matched otherwise simple exclusion from candidate list.
            for k, v in self.correct_letters.items():
                if word[k] != v:
                    if self.limit_answer:
                        if word_index in self.reduced_index_mapping:
                            self.valid_indexes[self.reduced_index_mapping[word_index]] = 0
                    else:
                        self.valid_indexes[word_index] = 0
                    inc = False
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
                        if self.limit_answer:
                            if word_index in self.reduced_index_mapping:
                                self.valid_indexes[self.reduced_index_mapping[word_index]] = 0
                        else:
                            self.valid_indexes[word_index] = 0
                        inc = False
                        break
            for k,v in self.min_limits.items():
                if k not in word_char_map.keys():
                    if self.limit_answer:
                        if word_index in self.reduced_index_mapping:
                            self.valid_indexes[self.reduced_index_mapping[word_index]] = 0
                    else:
                        self.valid_indexes[word_index] = 0
                    inc = False
                    break
                if word_char_map[k] < v:
                    if self.limit_answer:
                        if word_index in self.reduced_index_mapping:
                            self.valid_indexes[self.reduced_index_mapping[word_index]] = 0
                    else:
                        self.valid_indexes[word_index] = 0
                    inc = False
                    break
            if not inc:
                continue
            # lastly, check if an existing letter is in a position we know to be incorrect.
            for i in range(len(word)):
                if i in skip_indexes:
                    continue
                if word[i] in self.chars_incorrect_pos.keys():
                    if i in self.chars_incorrect_pos[word[i]]:
                        if self.limit_answer:
                            if word_index in self.reduced_index_mapping:
                                self.valid_indexes[self.reduced_index_mapping[word_index]] = 0
                        else:
                            self.valid_indexes[word_index] = 0
                        inc = False
                        break
            if not inc:
                continue
            updated_valid_word_list.append(word)
        self.valid_word_list = updated_valid_word_list
        new_length = len(self.valid_word_list)
        return

    def update_state(self, feedback: List[str]):
        self.feedback_history.append(feedback.copy())
        self.valid_word_list = [
            word for word in self.valid_word_list
            if matches_feedback_history(word, self.guess_record, self.feedback_history)
        ]
        possible_solutions = self.valid_solution_list if self.limit_answer else self.all_valid_word_list
        self.valid_indexes = np.asarray([
            matches_feedback_history(word, self.guess_record, self.feedback_history)
            for word in possible_solutions
        ], dtype=bool)
        return

    @staticmethod
    def decode(encoding_integer: int) -> str:
        pattern = ""
        while encoding_integer > 0:
            rem = encoding_integer % 3
            if rem == 2:
                pattern += 'G'
            elif rem == 1:
                pattern += 'Y'
            else:
                pattern += 'X'
            encoding_integer //= 3
        return pattern

    @staticmethod
    def count(pattern: str | None, update_dict: Dict[str, int]):
        if pattern is not None:
            update_dict[pattern] = update_dict.get(pattern, 0) + 1
        return

    # for testing, a conventional function that replicates the vectorized functionality.
    def calcs(self, word_idx_arr: np.ndarray, valid_index: np.ndarray, pattern_counts: Dict[str, int]) -> None:
        for encoding_num in word_idx_arr:
            if valid_index[encoding_num] == 0:
                continue
            decoded_encoding = self.decode(encoding_num)
            pattern_counts[decoded_encoding] = pattern_counts.get(decoded_encoding, 0) + 1
        return

    @staticmethod
    def load_precomputed_guess() -> Optional[pd.DataFrame]:
        try:
            guess_df = pd.read_csv('first_10_guesses.csv')
        except FileNotFoundError:
            return None
        return guess_df

    def _weight_total_entropy(self, entropy_val: float, word_key: str) -> float:
        if not self.use_frequencies or self.guess_count < 3:
            return entropy_val
        try:
            freq = float(self.frequency_df.loc[word_key, "frequency"])
        except KeyError:
            return 10e-5
        frequency_exponent = 1.0 if self.guess_count == 3 else 1.5
        return entropy_val * freq ** frequency_exponent

    def _rank_entropy(self, entropy_val: float, word_key: str) -> float:
        score = self._weight_total_entropy(entropy_val, word_key)
        if self.relative_frequency_weight:
            try:
                word_frequency = float(self.frequency_df.loc[word_key, "count"])
            except KeyError:
                word_frequency = 0.0
            relative_frequency = word_frequency / self.max_frequency if self.max_frequency > 0 else 0.0
            score += self.relative_frequency_weight * relative_frequency
        return score

    def next_guess(self):
        if self.guess_count == 1:
            cache_key = (self.limit_answer, self.use_frequencies, self.relative_frequency_weight)
            first_guess = _FIRST_GUESS_CACHE.get(cache_key)
            if first_guess is None:
                if self.relative_frequency_weight > 0:
                    ranked_guesses = self.top_10_guesses()
                    if not ranked_guesses:
                        raise RuntimeError("No valid guesses remain.")
                    first_guess = ranked_guesses[0][0]
                else:
                    guess_df = self.load_precomputed_guess()
                    if guess_df is None:
                        ranked_guesses = self.top_10_guesses()
                        if not ranked_guesses:
                            raise RuntimeError("No valid guesses remain.")
                        first_guess = ranked_guesses[0][0]
                    else:
                        first_guess = guess_df.sort_values(by='entropy', ascending=False).iloc[0]["guess"].strip()
                _FIRST_GUESS_CACHE[cache_key] = first_guess
            self.guess_count += 1
            self.guess_record.append(first_guess)
            return first_guess
        max_entropy, max_entropy_guess = -1.0, None
        t_1, t_2, t_c = 0.0, 0.0, 0.0
        for word in self.valid_word_list:
            idx = self.word_index_mapping[word]
            word_arr = self.guess_matrix[idx]
            ind = self.valid_indexes
            word_arr = word_arr[ind == 1]
            u_counts = np.unique_counts(word_arr).counts
            total_entropy, total_word_count = 0.0, sum(u_counts)
            for cnt in u_counts:
                p_s = cnt / total_word_count
                entropy = p_s * np.log2(p_s) * -1
                total_entropy += entropy
            total_entropy = self._rank_entropy(total_entropy, word)
            if total_entropy > max_entropy:
                max_entropy = total_entropy
                max_entropy_guess = word
        if max_entropy_guess is None:
            raise RuntimeError("No valid guesses remain.")
        self.guess_count += 1
        self.guess_record.append(max_entropy_guess)
        return max_entropy_guess

    def top_10_guesses(self):
        top_10_guesses: List[Tuple[str, float]] = []
        if not self.valid_word_list or not np.any(self.valid_indexes):
            raise RuntimeError("No valid guesses or possible solutions remain.")
        progress_bar = tqdm(desc="Guesses:", total=len(self.valid_word_list))
        for word in self.valid_word_list:
            idx = self.word_index_mapping[word]
            word_arr = self.guess_matrix[idx]
            v_decode, v_count = (np.vectorize(self.decode), np.vectorize(self.count, excluded={1,'update_dict'}))
            pattern_counts = {}
            ind = self.valid_indexes
            word_arr = word_arr[ind == 1]
            v_count(v_decode(word_arr), pattern_counts)
            total_entropy, total_word_count = 0.0, sum(v for v in pattern_counts.values())
            for k, v in pattern_counts.items():
                p_s = v / total_word_count
                entropy = p_s * np.log2(p_s) * -1
                total_entropy += entropy
            total_entropy = self._rank_entropy(total_entropy, word)
            min_entropy = min(top_10_guesses, key=lambda x: x[1])[1] if top_10_guesses else float('-inf')
            if total_entropy > min_entropy or len(top_10_guesses) < 10:
                top_10_guesses.append((word, total_entropy))
                top_10_guesses.sort(key=lambda x: x[1], reverse=True)
                top_10_guesses = top_10_guesses[:10]
            progress_bar.update(1)
        progress_bar.close()
        return top_10_guesses


if __name__ == "__main__":
    # solver = EntropySolver()
    # for i in range(10):
    #     print(i, "---------------------------------------")
    #     t1 = time.time()
    #     x = solver.next_guess()
    #     print("Time to calculate guess:", time.time() - t1)
    #     feedback = _calculate_pattern(x, "abase")
    #     tx = time.time()
    #     solver.update_state(feedback)
    #     print("abase", x, feedback)
    #     t2 = time.time()
    #     print("Time to update guess:", t2 - tx)
    #     print("Total time: ", t2 - t1)
    generate_guess_matrix()