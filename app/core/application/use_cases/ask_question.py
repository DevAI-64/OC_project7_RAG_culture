from dataclasses import dataclass
from typing import Protocol


class QuestionAnswerer(Protocol):
    """Port for a service capable of answering a user question."""

    def answer(self, question: str) -> str:
        """Generate an answer to a user question.

        Args:
            question: User's question.

        Returns:
            A natural-language answer.
        """


@dataclass(frozen=True)
class AskQuestionResult:
    """Result returned by the ask-question use case."""

    question: str
    answer: str


class AskQuestion:
    """Application use case for answering a user's question."""

    def __init__(self, answerer: QuestionAnswerer) -> None:
        """Initialize the use case.

        Args:
            answerer: Service responsible for generating the answer.
        """
        self._answerer = answerer

    def execute(self, question: str) -> AskQuestionResult:
        """Answer a user question.

        Args:
            question: User's question.

        Returns:
            The normalized question and its generated answer.

        Raises:
            ValueError: If the question is empty or contains only whitespace.
        """
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("La question ne peut pas être vide.")

        answer = self._answerer.answer(normalized_question)

        return AskQuestionResult(
            question=normalized_question,
            answer=answer,
        )
