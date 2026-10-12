# Pau Culture

Chatbot culturel intelligent permettant de rechercher des événements à Pau à partir de questions formulées en langage naturel.

L'application s'appuie sur les événements publiés dans OpenAgenda et combine :

- une ingestion des données depuis l'API Opendatasoft/OpenAgenda ;
- une normalisation et une structuration des événements ;
- une génération de descriptions adaptées à la recherche sémantique ;
- des embeddings Mistral ;
- un index vectoriel FAISS ;
- un filtrage déterministe des critères explicites ;
- une recherche sémantique sur les événements éligibles ;
- un LLM Mistral pour générer la réponse finale ;
- une API REST développée avec FastAPI.

Le projet a été réalisé dans le cadre du parcours **AI Engineer d'OpenClassrooms**.

---

## Sommaire

- [Fonctionnement](#fonctionnement)
- [Architecture](#architecture)
- [Pipeline d'ingestion](#pipeline-dingestion)
- [Pipeline RAG](#pipeline-rag)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Lancer l'application](#lancer-lapplication)
- [Utilisation de l'API](#utilisation-de-lapi)
- [Reconstruction des données et de l'index](#reconstruction-des-données-et-de-lindex)
- [Tests](#tests)
- [Docker](#docker)
- [Choix techniques](#choix-techniques)
- [Limites connues](#limites-connues)

---

## Fonctionnement

L'utilisateur peut poser des questions en langage naturel, par exemple :

> Quelles activités autour de la céramique sont prévues en octobre pour les enfants à moins de 10 € ?

Le système identifie les différents types de critères présents dans la question :

- **période** : octobre ;
- **public** : enfants ;
- **prix maximal** : moins de 10 € ;
- **thème sémantique** : céramique.

Les critères structurés sont appliqués de manière déterministe avant la recherche vectorielle.

Le thème sémantique est ensuite utilisé pour rechercher les événements les plus pertinents parmi les événements ayant déjà satisfait les critères explicites.

Enfin, les événements retenus sont transmis au modèle Mistral afin de générer une réponse naturelle en français.

### Principe général

```text
Question utilisateur
        │
        ▼
┌───────────────────┐
│   QueryParser     │
│                   │
│ période           │
│ prix              │
│ public            │
│ accessibilité     │
│ thème sémantique  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    EventFilter    │
│                   │
│ critères stricts  │
└─────────┬─────────┘
          │
          │ événements éligibles
          ▼
┌───────────────────┐
│  EventRetriever   │
│                   │
│ recherche FAISS   │
│ + embeddings      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  AnswerGenerator  │
│                   │
│      Mistral      │
└─────────┬─────────┘
          │
          ▼
    Réponse utilisateur
```

---

## Architecture

Le projet suit une organisation inspirée de la **Clean Architecture / Architecture Hexagonale**, avec une séparation entre :

- le domaine ;
- les cas d'utilisation ;
- l'infrastructure RAG et LLM ;
- l'ingestion des données ;
- les repositories ;
- l'API FastAPI.

```text
app/
├── main.py
│
├── core/
│   └── application/
│       └── use_cases/
│           └── ask_question.py
│
├── domain/
│   └── event.py
│
├── infrastructure/
│   ├── llm/
│   │   └── event_summarizer.py
│   │
│   └── rag/
│       ├── answer_generator.py
│       ├── event_filter.py
│       ├── event_retriever.py
│       ├── query_parser.py
│       └── rag_question_answerer.py
│
├── ingestion/
│   ├── document_builder.py
│   ├── event_processor.py
│   ├── faiss_indexer.py
│   ├── openagenda_client.py
│   └── pipeline.py
│
└── repositories/
    ├── event_catalog_repository.py
    └── json_snapshot_repository.py
```

### Responsabilités principales

| Composant | Responsabilité |
|---|---|
| `Event` | Modèle métier d'un événement |
| `AskQuestion` | Cas d'utilisation de réponse à une question |
| `QueryParser` | Extraction des critères explicites |
| `EventFilter` | Filtrage déterministe |
| `EventRetriever` | Recherche sémantique dans FAISS |
| `AnswerGenerator` | Génération de la réponse avec Mistral |
| `OpenAgendaClient` | Récupération des événements |
| `EventProcessor` | Normalisation des données brutes |
| `DocumentBuilder` | Construction des documents destinés au RAG |
| `EventSummarizer` | Génération de descriptions synthétiques |
| `FaissIndexer` | Création et persistance de l'index vectoriel |
| `EventCatalogRepository` | Accès au catalogue structuré |
| `IngestionPipeline` | Orchestration de l'ingestion |

---

## Pipeline d'ingestion

Les événements sont récupérés depuis le dataset public OpenAgenda via l'API Opendatasoft.

L'application cible :

- la ville de **Pau** ;
- les événements dont la date de fin est située dans une fenêtre commençant à **J-365 jours** ;
- les événements disponibles dans les données OpenAgenda au moment de l'ingestion.

### Étapes

```text
OpenAgenda / Opendatasoft
          │
          ▼
OpenAgendaClient
          │
          ▼
EventProcessor
          │
          ▼
Event
          │
          ▼
DocumentBuilder
          │
          ▼
documents JSON
          │
          ▼
FaissIndexer
          │
          ▼
Embeddings Mistral
          │
          ▼
Index FAISS
```

Deux snapshots sont conservés :

```text
data/
├── raw/
│   └── openagenda/
│       └── YYYY_MM_DD_events.json
│
└── processed/
    ├── openagenda/
    │   └── YYYY_MM_DD_documents.json
    │
    └── faiss_index/
        ├── index.faiss
        └── index.pkl
```

Les snapshots permettent notamment de conserver une trace des données utilisées pour construire l'index.

---

## Pipeline RAG

Le pipeline RAG est organisé en deux phases distinctes.

### 1. Filtrage déterministe

Les contraintes explicites sont extraites par `QueryParser`.

Exemples :

| Question | Critère extrait |
|---|---|
| `en octobre` | période |
| `moins de 10 €` | prix maximum |
| `pour les enfants` | public |
| `accessible PMR` | accessibilité |

Ces critères sont ensuite appliqués par `EventFilter`.

Cette étape est volontairement réalisée **avant** la recherche vectorielle.

Par exemple, un événement très proche sémantiquement de « céramique » mais coûtant 25 € ne doit pas être retourné pour une question demandant des événements à moins de 10 €.

### 2. Recherche sémantique

Lorsqu'un thème est identifié, seuls les événements ayant passé les filtres déterministes sont soumis à FAISS.

```text
Catalogue complet
       │
       ▼
Filtres explicites
       │
       ├── période
       ├── prix
       ├── public
       └── accessibilité
       │
       ▼
Événements éligibles
       │
       ▼
Recherche sémantique FAISS
       │
       ▼
Événements pertinents
       │
       ▼
Mistral
       │
       ▼
Réponse
```

Lorsque la question ne contient pas de thème sémantique, le système ne réalise pas de recherche vectorielle inutile et travaille directement avec les événements ayant satisfait les critères structurés.

---

## Structure du projet

```text
Project7/
│
├── app/
│   ├── core/
│   ├── domain/
│   ├── infrastructure/
│   ├── ingestion/
│   ├── repositories/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── openagenda/
│   └── processed/
│       ├── openagenda/
│       └── faiss_index/
│
├── tests/
│
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .env.example
└── README.md
```

---

## Prérequis

- Python **3.14+**
- [uv](https://docs.astral.sh/uv/)
- Une clé API Mistral
- Docker, uniquement pour l'exécution conteneurisée

---

## Installation

Cloner le projet puis se placer dans son répertoire :

```bash
cd Project7
```

Installer les dépendances avec `uv` :

```bash
uv sync
```

Les dépendances de développement sont également installées :

- pytest ;
- pytest-cov ;
- pytest-asyncio ;
- Ruff ;
- Black ;
- mypy.

---

## Configuration

Créer un fichier `.env` à partir du modèle fourni :

```bash
cp .env.example .env
```

Puis renseigner la clé API Mistral :

```env
MISTRAL_API_KEY=your_mistral_api_key
```

Le fichier `.env` ne doit pas être versionné.

La clé est utilisée notamment pour :

- les embeddings avec `mistral-embed` ;
- la génération de descriptions d'événements ;
- la génération des réponses du chatbot.

---

## Lancer l'application

L'application peut être lancée avec Uvicorn :

```bash
uv run uvicorn app.main:app --reload
```

L'API est alors disponible à :

```text
http://localhost:8000
```

La documentation interactive FastAPI est disponible à :

```text
http://localhost:8000/docs
```

---

## Utilisation de l'API

### Vérifier l'état de l'application

```http
GET /health
```

Exemple :

```bash
curl http://localhost:8000/health
```

### Poser une question

```http
POST /ask
```

Corps de la requête :

```json
{
  "question": "Quelles activités autour de la céramique sont prévues en octobre pour les enfants à moins de 10 € ?"
}
```

Exemple avec `curl` :

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Quelles activités autour de la céramique sont prévues en octobre pour les enfants à moins de 10 € ?\"}"
```

Réponse :

```json
{
  "question": "Quelles activités autour de la céramique sont prévues en octobre pour les enfants à moins de 10 € ?",
  "answer": "..."
}
```

---

## Reconstruction des données et de l'index

L'endpoint :

```http
POST /rebuild
```

permet de reconstruire le catalogue et l'index vectoriel.

Le pipeline :

1. récupère les événements OpenAgenda ;
2. sauvegarde le snapshot brut ;
3. transforme les événements ;
4. génère le catalogue traité ;
5. génère les embeddings ;
6. reconstruit l'index FAISS ;
7. recharge le catalogue et l'index dans l'application.

Exemple :

```bash
curl -X POST http://localhost:8000/rebuild
```

L'index existant est conservé tant que la nouvelle construction n'a pas abouti.

---

## Tests

Lancer l'ensemble des tests :

```bash
uv run pytest
```

Lancer les tests avec la couverture :

```bash
uv run pytest --cov=app --cov-report=term-missing
```

La configuration du projet vise une couverture minimale de **90 %**.

Les tests couvrent notamment :

- le domaine ;
- le parsing des questions ;
- le filtrage des événements ;
- la recherche sémantique ;
- la génération de documents ;
- l'ingestion ;
- la construction de l'index FAISS ;
- les repositories ;
- les endpoints FastAPI.

---

## Docker

Construire l'image :

```bash
docker build -t pau-culture .
```

Lancer le conteneur :

```bash
docker run --env-file .env -p 8000:8000 pau-culture
```

L'API est alors accessible à :

```text
http://localhost:8000
```

### Gestion des secrets

La clé Mistral n'est pas intégrée à l'image Docker.

Elle est fournie au démarrage du conteneur avec :

```bash
--env-file .env
```

Le fichier `.env` doit rester local et ne doit pas être ajouté au dépôt Git.

---

## Choix techniques

### FastAPI

FastAPI fournit l'API HTTP du chatbot et permet notamment de disposer automatiquement d'une documentation OpenAPI interactive.

### Mistral

Mistral est utilisé à plusieurs étapes :

- `mistral-embed` pour les embeddings ;
- le modèle configuré par l'application pour la génération de descriptions et de réponses.

### FAISS

FAISS est utilisé comme moteur de recherche vectorielle local.

Les documents sont découpés en fragments avant leur vectorisation afin de conserver une granularité adaptée à la recherche sémantique.

### Filtrage déterministe avant recherche sémantique

Le choix de filtrer les critères explicites avant la recherche vectorielle permet de distinguer deux types de contraintes :

**Contraintes déterministes**

- date ;
- prix ;
- public ;
- accessibilité.

**Recherche sémantique**

- thème ;
- activité ;
- formulation ou intention proche.

Cette séparation permet notamment d'éviter qu'une proximité sémantique compense une contrainte explicite non respectée.

---

## Limites connues

### Occurrences des événements récurrents

Le filtrage temporel utilise actuellement les dates `first_date` et `last_date`.

Pour un événement récurrent, cela permet de déterminer qu'un événement appartient globalement à une période, mais ne permet pas encore de vérifier précisément qu'une occurrence existe à une date donnée.

Une évolution future consiste à exploiter directement les occurrences contenues dans `timings`.

### Dépendance aux données OpenAgenda

Les réponses dépendent des événements disponibles dans OpenAgenda au moment de la dernière ingestion.

Un événement absent des données sources ne peut pas être retourné par le chatbot.

### Dépendance à l'API Mistral

La génération des embeddings et des réponses nécessite une clé API Mistral et un accès au service.

---

## Évolutions possibles

Plusieurs améliorations peuvent être envisagées :

- filtrage précis des occurrences des événements récurrents ;
- amélioration de l'extraction des critères en langage naturel ;
- enrichissement des critères de recherche ;
- meilleure présentation des informations selon la question posée ;
- ajout de liens directs vers les événements ;
- mise en place d'une stratégie de rafraîchissement automatique des données ;
- remplacement progressif de `langchain-community` par les intégrations dédiées lorsque cela sera nécessaire.

---

## Licence

Projet réalisé dans le cadre de la formation **AI Engineer – OpenClassrooms**.
