import os
import pytest


@pytest.fixture
def default_config_path(tmp_path):
    """Write default.yaml content to tmp_path and return the path."""
    yaml_content = """\
project:
  name: "regulatory-ref-extraction"
  seed: 42

device:
  auto_detect: true

model:
  name: "deepset/gbert-large"
  use_crf: false
  freeze_backbone: false
  use_lora: false
  lora_rank: 16

training:
  batch_size: 4
  learning_rate_backbone: 2.0e-5
  learning_rate_head: 1.0e-4
  warmup_steps: 100
  max_grad_norm: 1.0
  num_epochs: 3
  mixed_precision: "bf16"

data:
  max_seq_length: 512
  samples_per_batch: 8
  negative_sample_ratio: 0.4
  cache_dir: "data/cache"
  gold_test_dir: "data/gold_test"
  llm_seed: 1337
  llm_model: "google/gemini-flash-1.5"

ensemble:
  enabled: false
  n_estimators: 3

evaluation:
  output_dir: "evaluation_output"
"""
    config_file = tmp_path / "default.yaml"
    config_file.write_text(yaml_content)
    return str(config_file)


# ---------------------------------------------------------------------------
# German legal reference samples — fixtures for regex baseline & metrics tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_paragraph_ref():
    """Simple paragraph reference: § 25a KWG."""
    text = "Gemäß § 25a KWG gilt"
    expected_text = "§ 25a KWG"
    start = text.index(expected_text)
    end = start + len(expected_text)
    return {"text": text, "spans": [(start, end)], "expected_texts": [expected_text]}


@pytest.fixture
def sample_paragraph_with_absatz():
    """Paragraph with Absatz: § 25a Abs. 1 KWG."""
    text = "§ 25a Abs. 1 KWG"
    return {"text": text, "spans": [(0, len(text))], "expected_texts": [text]}


@pytest.fixture
def sample_artikel():
    """Artikel reference: Art. 6 DSGVO."""
    text = "Art. 6 DSGVO regelt"
    expected_text = "Art. 6 DSGVO"
    start = text.index(expected_text)
    end = start + len(expected_text)
    return {"text": text, "spans": [(start, end)], "expected_texts": [expected_text]}


@pytest.fixture
def sample_anhang():
    """Anhang reference: Anhang II CRR."""
    text = "Anhang II CRR enthält"
    expected_text = "Anhang II CRR"
    start = text.index(expected_text)
    end = start + len(expected_text)
    return {"text": text, "spans": [(start, end)], "expected_texts": [expected_text]}


@pytest.fixture
def sample_verordnung():
    """EU-Verordnung reference: EU-Verordnung 648/2012."""
    text = "EU-Verordnung 648/2012"
    return {"text": text, "spans": [(0, len(text))], "expected_texts": [text]}


@pytest.fixture
def sample_multi_paragraph():
    """Multi-section §§: §§ 3, 4 UWG."""
    text = "§§ 3, 4 UWG"
    return {"text": text, "spans": [(0, len(text))], "expected_texts": [text]}


@pytest.fixture
def sample_satz():
    """Satz reference: § 5 Abs. 2 S. 1 BGB."""
    text = "§ 5 Abs. 2 S. 1 BGB"
    return {"text": text, "spans": [(0, len(text))], "expected_texts": [text]}


@pytest.fixture
def sample_nr():
    """Nr reference: § 1 Nr. 3 KWG."""
    text = "§ 1 Nr. 3 KWG"
    return {"text": text, "spans": [(0, len(text))], "expected_texts": [text]}


@pytest.fixture
def sample_lit():
    """Litera reference: § 2 Abs. 1 lit. a BGB."""
    text = "§ 2 Abs. 1 lit. a BGB"
    return {"text": text, "spans": [(0, len(text))], "expected_texts": [text]}


@pytest.fixture
def sample_tz():
    """Teilziffer reference: Tz. 4 MaRisk."""
    text = "Tz. 4 MaRisk"
    return {"text": text, "spans": [(0, len(text))], "expected_texts": [text]}


@pytest.fixture
def sample_no_reference():
    """Text with no legal references."""
    text = "Dies ist ein normaler Satz."
    return {"text": text, "spans": [], "expected_texts": []}
