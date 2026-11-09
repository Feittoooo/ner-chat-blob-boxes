from ner_chat.ask import run_ask
from ner_chat.store import EntityStore
from pathlib import Path
from pydantic_ai.models.test import TestModel
from pydantic_ai import models

def test_known_and_absent_names():
    models.ALLOW_MODEL_REQUESTS = False

    model = TestModel(custom_output_args= {
    "reply": "<text>",
    "entities": [
        {"name": "Portugal", "label": "<whatever>", "mention_count": 999},
        {"name": "California", "label": "<whatever>", "mention_count":32},
    ],
    })

    store_path = Path(__file__).parent / "entities.json"
    store = EntityStore.load(store_path)
    response = run_ask("In which country is Lisboa", store, model)

    entries = {e.name: e for e in response.entities}

    assert entries["Portugal"].mention_count == 7
    assert entries["Portugal"].label == "country"
    assert entries["California"].mention_count == 0
    assert entries["California"].label == "not mentioned"
