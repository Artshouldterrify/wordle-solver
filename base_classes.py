import time
from typing import Optional, Any, Dict, List
import numpy as np
from tqdm import tqdm
from wordle_feedback import calculate_wordle_feedback
from entropy_solver import EntropySolver
from stubh_solver import StubhSolver
from llm_solver import LLMSolver
from statistical_solver import StatisticalSolver
from statistic_tracker import StatisticTracker


# class for simulating a single game.
class WordleGame:
    def __init__(self, solution_word: str, valid_guess_list: List[str], max_guesses: int = 8, statistic_tracker: Optional[Any] = None):
        self._solve_word = solution_word
        self._valid_guess_list = valid_guess_list
        self._current_guess_count, self._max_guesses = 0, max_guesses
        self._statistic_tracker = statistic_tracker
        self._guess_list = []

    def __map(self) -> Dict[str, int]:
        char_map = {}
        for char in self._solve_word:
            char_map[char] = char_map.get(char, 0) + 1
        return char_map

    # the comparison function.
    def __comparison_function(self, guess: str) -> List[str] | bool:
        solution_map = self.__map()
        feedback, solved = [], True
        for i in range(len(self._solve_word)):
            if guess[i] == self._solve_word[i]:
                feedback.append('G')
                solution_map[guess[i]] -= 1
            else:
                feedback.append('X')
                solved = False
        for i in range(len(self._solve_word)):
            if feedback[i] == 'X':
                if guess[i] in solution_map and solution_map[guess[i]] > 0:
                    feedback[i] = 'Y'
                    solved = False
                    solution_map[guess[i]] -= 1
        if solved:
            return True
        return feedback

    def __handle_game_over(self, guess_list: List[str], was_solved: bool) -> None:
        if self._statistic_tracker is not None:
            self._statistic_tracker.add_record(was_solved=was_solved, solution_word=self._solve_word, n_guesses=self._current_guess_count, guess_list=guess_list)
        return

    # solver game over function, used when the solver gives up.
    def solver_game_over(self, guess_list: List[str]) -> None:
        if self._statistic_tracker is not None:
            solved = False
            self._statistic_tracker.add_record(was_solved=solved, solution_word=self._solve_word, n_guesses=len(guess_list), guess_list=guess_list)
        return

    # handle specifics of making a guess.
    def make_guess(self, guess: str) -> List[str] | bool:
        if guess not in self._valid_guess_list:
            return False
        feedback = self.__comparison_function(guess)
        self._current_guess_count += 1
        self._guess_list.append(guess)
        if feedback is True or self._current_guess_count >= self._max_guesses:
            self.__handle_game_over(guess_list=self._guess_list, was_solved=feedback is True)
            return True
        return feedback


# Baseline solver for testing runtime.
class BaselineSolver:
    def __init__(self):
        self.guess_word = "crane"

    def reset(self):
        pass

    def next_guess(self) -> str:
        return self.guess_word

    def update_state(self, feedback: List[str]):
        pass


# Base statistic tracker.
class BaseStatisticsTracker:
    def __init__(self):
        self.data: List[Dict[str, Any]] = []

    # required method, add record of a single game.
    def add_record(self, was_solved: bool, solution_word: str, n_guesses: int, guess_list: Optional[List[str]] = None) -> None:
        if guess_list is None:
            guess_list = []
        self.data.append({
            'was_solved': was_solved,
            'solution_word': solution_word,
            'n_guesses': n_guesses,
            'guess_list': guess_list
        })
        return

    # reset the tracker. Used when running a new simulation.
    def reset(self) -> None:
        self.data = []
        return

    def __percentage_solved(self) -> float:
        solved, total = 0, 0
        for record in self.data:
            if record['was_solved']:
                solved += 1
            total += 1
        return solved / total if total else 0

    def __average_guess_score(self) -> float:
        guess_score, total = 0, 0
        for record in self.data:
            if record['was_solved']:
                guess_score += record['n_guesses']
                total += 1
        return guess_score / total if total > 0 else 0

    def summary(self) -> str:
        summary = ""
        summary += "Percentage Solved: " + str(self.__percentage_solved()) + "\n"
        summary += "Average Guess Score: " + str(self.__average_guess_score()) + "\n"
        return summary


# class for handling simulations.
class WordleInterface:
    def __init__(self, statistic_tracker: Optional[Any] = None):
        self.guess_list = self._build_valid_guess_list()
        self.solution_list = self._build_valid_solution_list()
        if statistic_tracker is not None:
            self.statistic_tracker = statistic_tracker
        else:
            self.statistic_tracker = StatisticTracker()

    @staticmethod
    def _pick_solution_word() -> str:
        answers = []
        with open('wordle_answers.txt', 'r') as f:
            for line in f.readlines():
                answers.append(line.strip())
        rng = np.random.default_rng()
        solution_word = rng.choice(answers)
        return solution_word

    @staticmethod
    def _build_valid_guess_list() -> List[str]:
        guesses = []
        with open('wordle_answers.txt', 'r') as f:
            for line in f.readlines():
                guesses.append(line.strip())
        with open('wordle_guesses.txt', 'r') as f:
            for line in f.readlines():
                guesses.append(line.strip())
        return guesses

    @staticmethod
    def _build_valid_solution_list() -> List[str]:
        sols = []
        with open('wordle_answers.txt', 'r') as f:
            for line in f.readlines():
                sols.append(line.strip())
        return sols

    # main simulation function.
    def simulate_game(self, wordle_solver: Any, n_games: int = 10, max_guesses: int = 8) -> None:
        try:
            self.statistic_tracker.reset()
            progress_bar = tqdm(total=n_games, desc="Simulating game: ")
            i, errors = 0, 0
            while i < n_games:
                if errors > 10:
                    raise RuntimeError("Too many errors in simulation. Exiting.")
                wordle_solver.reset()
                solution_word = self._pick_solution_word()
                print("Game " + str(i+1) + ": Solution word is: " + solution_word)
                game_instance = WordleGame(solution_word, self.guess_list, statistic_tracker=self.statistic_tracker, max_guesses=max_guesses)
                error_exit = False
                while True:
                    guess_word = wordle_solver.next_guess()
                    if guess_word == "give up":
                        game_instance.solver_game_over(wordle_solver.guess_history)
                        break
                    if not guess_word:
                        error_exit = True
                        break
                    feedback = game_instance.make_guess(guess_word)
                    if feedback is True:
                        break
                    if feedback is False:
                        error_exit = True
                        break
                    wordle_solver.update_state(feedback)
                if error_exit:
                    errors += 1
                    game_instance.solver_game_over(wordle_solver.guess_history)
                i += 1
                progress_bar.update(1)
            progress_bar.close()
            return True
        except Exception as e:
            raise RuntimeError("Error in executing simulation: " + str(e))

    # test execution times for a 1000 games simulation using the baseline solver class.
    def simulation_test(self, solver_class: Any = None, n_games: int = 1000, file_name: Optional[str] = "BaselineSolver_SimulationTest.csv"):
        if solver_class is None:
            solver_instance = BaselineSolver()
        else:
            solver_instance = solver_class
        self.statistic_tracker.set_file_name(file_name)
        t1 = time.time()
        print("Starting simulation at time: " + str(t1))
        self.simulate_game(solver_instance, n_games)
        t2 = time.time()
        print("Finished simulation at time: " + str(t2))
        print("Time taken: " + str(t2 - t1) + " seconds")
        print(self.statistic_tracker.summary())
        return

    # standard simulation: a single pass over every single possible solution.
    def standard_metric_simulation(self, solver_class: Any = None, file_name: Optional[str] = "BaselineSolver_StandardSimulation.csv"):
        if solver_class is None:
            solver_instance = BaselineSolver()
        else:
            solver_instance = solver_class
        progress_bar = tqdm(total=len(self.solution_list), desc="Simulating games over every single possible solution word: ")
        self.statistic_tracker.reset()
        self.statistic_tracker.set_file_name(file_name)
        for word in self.solution_list:
            solver_instance.reset()
            solution_word = word
            game_instance = WordleGame(solution_word, self.guess_list, statistic_tracker=self.statistic_tracker, max_guesses=6)
            while True:
                guess_word = solver_instance.next_guess()
                feedback = game_instance.make_guess(guess_word)
                if feedback is True:
                    break
                solver_instance.update_state(feedback)
            progress_bar.update(1)
        progress_bar.close()
        print(self.statistic_tracker.summary())
        return True


if __name__ == "__main__":
    interface = WordleInterface()
    solvers = [StubhSolver()]
    solver_names = ["StubhSolver_SALET_StandardSimulation.csv"]
    for solver, name in zip(solvers, solver_names):
        interface.standard_metric_simulation(solver_class=solver, file_name=name)
    # llm_solver = LLMSolver("gemma4:e4b")
    # interface.simulation_test(solver_class=llm_solver, n_games=20, file_name="Gemma4_E4B_NOCOTLMSolver_SimulationTest.csv")
    # {'total_games': 50, 'solved_games': 43, 'unsolved_games': 7, 'avg_guesses': np.float64(4.88), 'percentage_solved': 86.0}
    # Statistical Solver:
    # {'total_games': 2315, 'solved_games': 2204, 'unsolved_games': 111, 'avg_guesses': np.float64(4.261343012704174), 'percentage_solved': 95.20518358531318}
    # Entropy Solver:
    # {'total_games': 2315, 'solved_games': 2231, 'unsolved_games': 84, 'avg_guesses': np.float64(4.0918870461676375), 'percentage_solved': 96.37149028077754}
    # Entropy Solver (limit answer):
    # {'total_games': 2315, 'solved_games': 2309, 'unsolved_games': 6, 'avg_guesses': np.float64(3.5695106106539627), 'percentage_solved': 99.74082073434126}
    # Stubh Solver:
    # {'total_games': 2315, 'solved_games': 2295, 'unsolved_games': 20, 'avg_guesses': np.float64(3.8113289760348583), 'percentage_solved': 99.13606911447084}