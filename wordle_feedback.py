from typing import List


def calculate_wordle_feedback(guess: str, solution: str) -> List[str]:
    if len(guess) != len(solution):
        raise ValueError("Guess and solution must have the same length.")
    feedback = ["X"] * len(guess)
    remaining = {}
    for letter in solution:
        remaining[letter] = remaining.get(letter, 0) + 1
    for position, letter in enumerate(guess):
        if letter == solution[position]:
            feedback[position] = "G"
            remaining[letter] -= 1
    for position, letter in enumerate(guess):
        if feedback[position] == "X" and remaining.get(letter, 0) > 0:
            feedback[position] = "Y"
            remaining[letter] -= 1
    return feedback


def matches_feedback_history(candidate: str, guesses: List[str], feedback_history: List[List[str]]) -> bool:
    return len(guesses) == len(feedback_history) and all(
        len(feedback) == len(guess) == len(candidate)
        and calculate_wordle_feedback(guess, candidate) == feedback
        for guess, feedback in zip(guesses, feedback_history)
    )