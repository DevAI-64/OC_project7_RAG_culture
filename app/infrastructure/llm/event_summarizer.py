import os

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


class EventSummarizer:
    """Generate concise event descriptions for semantic search."""

    SYSTEM_PROMPT = """
Tu es un assistant qui prépare des descriptions
d'événements culturels pour un moteur de recherche
sémantique.

À partir du texte fourni :
- Produis une description concise et naturelle.
- Conserve le titre, le type d'événement, son thème,
  ses activités et ses particularités.
- Conserve les informations distinctives utiles
  pour retrouver l'événement par une question.
- Supprime les répétitions et les formulations
  promotionnelles génériques.
- N'invente aucune information.
- N'utilise aucune connaissance extérieure.
- Ne déduis pas de public, d'âge, de prix,
  de date ou d'accessibilité.
- Si le texte est vide, retourne une chaîne vide.

Retourne uniquement le texte final.
"""

    def __init__(self, model: str) -> None:
        """Initialize the Mistral client used for summarization.

        Args:
            model: Mistral model identifier used for text generation.

        Raises:
            RuntimeError: If the ``MISTRAL_API_KEY`` environment variable
                is not configured.
        """
        api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            raise RuntimeError(
                "La variable d'environnement "
                "MISTRAL_API_KEY est absente. "
                "Vérifiez votre fichier .env."
            )

        self.client = Mistral(api_key=api_key)
        self.model = model

    def summarize(self, text: str) -> str:
        """Generate a concise semantic description of an event.

        Args:
            text: Event content to summarize.

        Returns:
            A concise description suitable for semantic embedding.

        Raises:
            ValueError: If Mistral returns empty or non-textual content.
            Exception: If the Mistral API request fails.
        """
        if not text.strip():
            return ""

        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        content = response.choices[0].message.content

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Le LLM a retourné un résumé vide.")

        return content.strip()
