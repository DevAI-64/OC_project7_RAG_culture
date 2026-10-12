from app.ingestion.pipeline import IngestionPipeline


def test_pipeline_runs_all_steps(monkeypatch) -> None:
    class Client:
        def fetch_events(self):
            return {"metadata": {"city": "Pau"}, "events": [{"uid": "1"}]}

    class Processor:
        def process(self, events):
            return ["event"]

    class Builder:
        def build(self, event):
            return {"id": event}

    class Repository:
        def __init__(self):
            self.saved = []

        def save(self, path, payload):
            self.saved.append((path, payload))

    repository = Repository()
    monkeypatch.setattr("app.ingestion.pipeline.date", __import__("datetime").date)
    result = IngestionPipeline(Client(), Processor(), Builder(), repository).run()
    assert result["raw_count"] == 1
    assert result["processed_count"] == 1
    assert result["document_count"] == 1
    assert len(repository.saved) == 2
    assert result["raw_path"].endswith("_events.json")
    assert result["processed_path"].endswith("_documents.json")
