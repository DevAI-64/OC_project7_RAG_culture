from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Tu es l'assistant du chatbot culturel de Pau.

Réponds en français, de manière claire, naturelle et concise.

Les événements fournis dans le contexte ont déjà été filtrés
par le programme selon les contraintes explicites de l'utilisateur.
Tu dois donc uniquement présenter les événements fournis.

Règles :
- Utilise exclusivement les informations du contexte.
- N'invente aucune information.
- Ne décide jamais qu'un événement satisfait une contrainte :
  cette vérification a déjà été effectuée par le programme.
- Ne transforme jamais une information inconnue en information positive.
- Si aucun événement ne correspond aux contraintes explicites, dis-le clairement.
- Ne mentionne que les informations utiles pour répondre à la question.
- N'ajoute pas systématiquement toutes les métadonnées disponibles.
- Le thème a été traité par recherche sémantique : les événements fournis
  peuvent donc être pertinents même si leurs mots ne correspondent pas
  exactement à ceux de la question.
- Ne complète jamais une information absente du contexte avec des connaissances
  externes.

CONTEXTE :
{context}
""",
        ),
        ("human", "{question}"),
    ]
)


class AnswerGenerator:
    """Generate a natural-language answer from filtered event results."""

    def __init__(self, model_name: str = "ministral-3b-latest") -> None:
        """Initialize the language model and answer generation chain.

        Args:
            model_name: Name of the Mistral model used to generate answers.
        """
        self._model = ChatMistralAI(
            model_name=model_name,
            temperature=0,
        )
        self._chain = PROMPT | self._model

    def generate(
        self,
        question: str,
        results: list[tuple[Document, float]],
    ) -> str:
        """Generate an answer from the semantically retrieved events.

        Args:
            question: Original question asked by the user.
            results: Events retrieved after applying the deterministic filters,
                together with their semantic relevance scores.

        Returns:
            A natural-language answer in French. If no event was retrieved,
            returns a message indicating that no matching event was found.
        """
        if not results:
            return (
                "Je n'ai pas trouvé d'événement correspondant à votre demande "
                "dans les données disponibles."
            )

        context = self._build_context(results)

        response = self._chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        return str(response.content).strip()

    @staticmethod
    def _build_context(
        results: list[tuple[Document, float]],
    ) -> str:
        """Build the textual context provided to the language model.

        Args:
            results: Retrieved events with their semantic relevance scores.

        Returns:
            Formatted event information suitable for the answer prompt.
        """
        context_parts: list[str] = []

        for index, (document, _score) in enumerate(results, start=1):
            metadata = document.metadata

            context_parts.append(
                f"""Événement {index}
Titre et description :
{document.page_content}

Informations structurées :
- Ville : {metadata.get("city", "Pau")}
- Adresse : {metadata.get("address", "Non renseignée")}
- Début : {metadata.get("first_date", "Non renseigné")}
- Fin : {metadata.get("last_date", "Non renseignée")}
- Tarif : {metadata.get("conditions", "Non renseigné")}
- Public : {", ".join(metadata.get("audience_categories", [])) or "Non renseigné"}
- Accessibilité : {
                    metadata.get("accessibility_label")
                    or metadata.get("accessibility")
                    or "Non renseignée"
                }
- URL : {metadata.get("url", "Non renseignée")}""".strip()
            )

        return "\n\n---\n\n".join(context_parts)
