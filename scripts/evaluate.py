"""CLI entry point for regex baseline evaluation.

Usage (from project root):
    PYTHONPATH=. python scripts/evaluate.py
    PYTHONPATH=. python scripts/evaluate.py project.seed=123
"""

from dotenv import load_dotenv

load_dotenv()  # Must be first — before any config or model imports

from src.utils.config import load_config
from src.utils.device import get_device, set_seed
from src.evaluation.evaluator import Evaluator


# Hardcoded demo samples with known reference spans.
# In Phase 4, this will read from data/gold_test/gold_test_set.json.
DEMO_SAMPLES = [
    {
        "text": "Gemäß § 25a KWG gilt eine besondere Organisationspflicht.",
        "spans": [(6, 15)],  # "§ 25a KWG"
    },
    {
        "text": "Art. 6 DSGVO regelt die Rechtmäßigkeit der Verarbeitung.",
        "spans": [(0, 12)],  # "Art. 6 DSGVO"
    },
    {
        "text": "Anhang II CRR enthält die Risikopositionen.",
        "spans": [(0, 13)],  # "Anhang II CRR"
    },
    {
        "text": "Nach § 1 Nr. 3 KWG ist dies ein Kreditinstitut.",
        "spans": [(5, 19)],  # "§ 1 Nr. 3 KWG"
    },
    {
        "text": "EU-Verordnung 648/2012 regelt die EMIR-Anforderungen.",
        "spans": [(0, 22)],  # "EU-Verordnung 648/2012"
    },
    {
        "text": "§ 25a Abs. 1 KWG fordert ein angemessenes Risikomanagement.",
        "spans": [(0, 17)],  # "§ 25a Abs. 1 KWG"
    },
    {
        "text": "Dies ist ein normaler Satz ohne Referenzen.",
        "spans": [],
    },
]


def main():
    cfg = load_config()
    device = get_device()
    set_seed(cfg.project.seed)

    print(f"Device: {device}")
    print(f"Seed: {cfg.project.seed}")
    print()

    evaluator = Evaluator(cfg)
    metrics = evaluator.evaluate_baseline(DEMO_SAMPLES)
    print(evaluator.format_report(metrics))


if __name__ == "__main__":
    main()
