import time

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import ToolCallLimitMiddleware
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver
import pandas as pd


def _build_valid_guess_list() -> List[str]:
    guesses = []
    with open('wordle_answers.txt', 'r') as f:
        for line in f.readlines():
            guesses.append(line.strip())
    with open('wordle_guesses.txt', 'r') as f:
        for line in f.readlines():
            guesses.append(line.strip())
    return guesses

valid_guesses = _build_valid_guess_list()

@tool(description="Check if a word is a valid guess.")
def is_valid_guess(word: str) -> bool:
    print("LLM checking if word is valid guess:", word)
    return word.lower().strip() in valid_guesses

class WordleResponse(BaseModel):
    """Class for the response from the LLM."""
    guess: str = Field(min_length=5, max_length=5, description="The guess word.")

class LLMSolver:
    def __init__(self, model_string: str = "nemotron-cascade-2:latest"):
        self.model_string = model_string
        self.model = ChatOllama(model=model_string, reasoning=False)
        self.feedback_history: List[List[str]] = []
        self.guess_history: List[str] = []
        self.guess_count = 1
        self.checkpointer = InMemorySaver()
        self.agent = create_agent(
            self.model,
            system_prompt=self._rules_prompt(),
            tools=[is_valid_guess],
            middleware=[ToolCallLimitMiddleware(run_limit=20)],
            checkpointer=self.checkpointer
        )
        self.usage_metadata = pd.DataFrame(columns=["guess_count","guess_list","input_tokens","output_tokens","total_tokens","reasoning_messages"])
        self.current_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_messages": []}
        return

    def reset(self):
        self.current_metadata["guess_count"] = self.guess_count
        self.current_metadata["guess_list"] = self.guess_history
        self.usage_metadata.loc[len(self.usage_metadata)] = self.current_metadata
        self.usage_metadata.to_csv("usage_metadata.csv", index=False)
        self.checkpointer = InMemorySaver()
        self.model = ChatOllama(model=self.model_string, reasoning=False)
        self.agent = create_agent(
            self.model,
            system_prompt=self._rules_prompt(),
            tools=[is_valid_guess],
            middleware=[ToolCallLimitMiddleware(run_limit=20)],
            checkpointer=self.checkpointer
        )
        self.feedback_history = []
        self.guess_history = []
        self.guess_count = 1
        self.current_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_messages": []}
        return

    @staticmethod
    def _rules_prompt() -> str:
        prompt = """
            You are playing a game of Wordle. You need to guess a 5-letter word. The feedback for each guess will be provided as a list of 5 characters, 
            where 'G' indicates a correct letter in the correct position, 'Y' indicates a correct letter in the wrong position, 
            and 'X' indicates an incorrect letter.
            
            Your task is to provide a valid 5-letter word as your next guess based on the feedback from previous guesses.
            
            DO NOT provide any explanations or reasoning in your response. Only provide the next guess as a single 5-letter word.
            DO NOT guess a word that is not in the valid guess list. You have a tool that can check if a word is valid. Use it if you are unsure.
            Most common words are sure to be in the valid guess list. 
            
            IMPORTANT: 
            DO NOT keep reasoning for a long time. If you are unsure of what to guess, make a guess anyway. 
            IF AT ANY POINT YOU ARE REASONING FOR OVER 10,000 TOKENS/A LONG TIME, YOU MUST MAKE A GUESS. DO NOT KEEP REASONING.
            
            IMPORTANT:
            if at any point you think you cannot make progress, you may output the string "give up" to indicate that you are giving up on the game. 
            You should only do this if you find you are reasoning for a long time and cannot make progress. 
            
            Your final response should be a valid 5-letter word that you want to guess next. Do not include any other text in your final guess response.
            No limits are enforced on any internal reasoning you do, but your final guess must be a valid 5-letter word.
        """
        return prompt

    def update_state(self, feedback: List[str]):
        self.feedback_history.append(feedback)
        return

    def _build_prompt(self) -> str:
        prompt = ""
        if self.guess_history:
            prompt += "Here are your previous guesses and their feedback:\n"
            for i, (guess, feedback) in enumerate(zip(self.guess_history, self.feedback_history)):
                prompt += f"Guess {i+1}: {guess} -> {feedback}\n"
        prompt += "Please provide your next guess as a 5-letter word."
        return prompt

    def next_guess(self) -> str:
        try:
            prompt = self._build_prompt()
            response = self.agent.invoke({"messages": [{"role": "user", "content": prompt}]}, {"configurable": {"thread_id": 1}})
            meta = response["messages"][-1].usage_metadata
            reasoning_messages = []
            for msg in response["messages"]:
                print(msg, "reasoning_content" in msg.additional_kwargs)
                if msg.additional_kwargs and "reasoning_content" in msg.additional_kwargs:
                    reasoning_messages.append(msg.additional_kwargs["reasoning_content"])
            response = response["messages"][-1].content.strip().lower()
            if response == "give up":
                print("LLM has given up.")
                return response
            while response not in valid_guesses:
                response = self.agent.invoke({"messages": [{"role": "user", "content": f"The word '{response}' is not a valid guess. Please provide a valid 5-letter word."}]}, {"configurable": {"thread_id": 1}})
                meta = response["messages"][-1].usage_metadata
                reasoning_messages = []
                for msg in response["messages"]:
                    if msg.additional_kwargs and "reasoning_content" in msg.additional_kwargs:
                        reasoning_messages.append(msg.additional_kwargs["reasoning_content"])
                response = response["messages"][-1].content.strip().lower()
                if response == "give up":
                    print("LLM has given up.")
                    return response
            print("LLM guess:", response)
            for k,v in meta.items():
                self.current_metadata[k] += v
            self.current_metadata["reasoning_messages"] += reasoning_messages
            print("Current metadata:", self.current_metadata)
            self.guess_history.append(response)
            self.guess_count += 1
            return response
        except Exception as e:
            print(e)
            return False