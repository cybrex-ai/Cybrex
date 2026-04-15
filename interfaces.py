from abc import ABC, abstractmethod
from typing import Iterator


class InputInterface(ABC):
    @abstractmethod
    def get_input(self) -> str:
        """Block until input is received. Returns input as string."""


class OutputInterface(ABC):
    @abstractmethod
    def send(self, token: str) -> None:
        """Receive a single token. Module handles buffering internally."""
    
    def interrupt(self) -> None:
        """Interrupt output generation"""


class CoreInterface(ABC):
    @abstractmethod
    def generate(self, data: str) -> Iterator[str]:
        """Takes user input and conversation context. Yields response tokens."""

    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None:
        """"""


class ShortTermMemoryInterface(ABC):
    @abstractmethod
    def add(self, role: str, content: str) -> None: pass
    
    @abstractmethod
    def get(self) -> list[dict]: pass
    
    @abstractmethod
    def clear(self) -> None: pass


class LongTermMemoryInterface(ABC):
    @abstractmethod
    def store(self, messages: list[dict]) -> None: pass
    
    @abstractmethod
    def retrieve(self, query: str) -> str: pass