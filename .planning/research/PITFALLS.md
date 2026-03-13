# Pitfalls Research: German Regulatory Reference Extraction

## Critical Pitfalls

### 1. Subword-to-Token BIO Alignment Bug
**Risk:** Silent label corruption from BERT subword tokenization. German legal terms get split into subwords (e.g., "Kreditwesengesetz" → ["Kredit", "##wesen", "##gesetz"]). If BIO labels aren't correctly propagated to subword tokens, the model trains on corrupted labels.
**Warning signs:** Model predicts partial spans; entity boundaries consistently off by 1-2 characters.
**Prevention:** Use `offset_mapping` from tokenizer. Test BIO conversion extensively with known examples before training. Verify round-trip: text → tokens → BIO labels → reconstructed spans == original spans.
**Phase:** Data pipeline (early phase)

### 2. LLM-Generated Data Quality Drift
**Risk:** Inflated F1 from LLM-to-LLM evaluation. If both training data AND gold test set come from the same LLM, the model learns LLM writing patterns, not real regulatory text patterns. Metrics look great but real-world performance suffers.
**Warning signs:** Very high F1 on LLM-generated test set but poor performance on real regulatory documents.
**Prevention:** Gold test set should be manually validated. Include real regulatory text samples if possible. Track distribution of reference types and text styles.
**Phase:** Evaluation setup

### 3. Entity-Level vs Token-Level Evaluation
**Risk:** Token-level F1 masks real extraction failures. A model that gets 95% of tokens right but consistently misses span boundaries has poor entity-level performance.
**Warning signs:** High token-level F1 but users report missing/wrong references.
**Prevention:** Always report entity-level metrics (exact match + partial match). Use seqeval library. Never rely solely on token-level accuracy.
**Phase:** Evaluation implementation

### 4. Special Token Label Masking
**Risk:** [CLS], [SEP], [PAD] tokens must get label -100 (ignored by loss). If labeled as O, the model wastes capacity learning to classify padding tokens and the loss signal is diluted.
**Warning signs:** Training loss plateaus at unexpectedly high value; model performance doesn't improve with more data.
**Prevention:** Explicitly set label=-100 for all special tokens. Verify in BIO converter unit tests.
**Phase:** Data pipeline

### 5. CRF Not Enforcing BIO Constraints
**Risk:** CRF learns transition probabilities from data rather than hard-coding valid BIO transitions. With LLM-generated data that may have label noise, the CRF might learn invalid transitions.
**Warning signs:** Model predicts I-REF after O; impossible label sequences in output.
**Prevention:** Initialize CRF transition matrix with hard constraints (O→I-REF = -inf). Validate decoded sequences post-inference.
**Phase:** Model implementation

### 6. gbert-large OOM on MPS + Wrong Learning Rate
**Risk:** gbert-large (1024d, ~335M params) can OOM on Apple M1 with 16GB unified memory. Using same learning rate for BERT and classifier head causes catastrophic forgetting of pretrained representations.
**Warning signs:** OOM errors on M1; loss spikes or doesn't converge.
**Prevention:** Differential learning rates (2e-5 for BERT, 1e-3 for head). Reduce batch_size for MPS. Implement gradient accumulation. Test with smaller batches first.
**Phase:** Training implementation

### 7. IterableDataset + Warmup Steps Mismatch
**Risk:** With IterableDataset (online LLM generation), total training steps are unknown upfront. Linear warmup scheduler needs total_steps. If estimated wrong, warmup is too short/long.
**Warning signs:** Learning rate schedule doesn't match expected behavior; training instability in early steps.
**Prevention:** Calculate total_steps from: epochs * estimated_batches_per_epoch. Or use a fixed warmup_steps instead of warmup_ratio. Log the actual LR at each step.
**Phase:** Training implementation

## Moderate Pitfalls

### 8. Regex Baseline Too Weak
**Risk:** A trivially weak regex baseline makes the ML model look good but doesn't prove value. If regex catches 90% of references, the ML model needs to beat that convincingly.
**Warning signs:** Regex baseline has surprisingly low scores on obviously capturable patterns.
**Prevention:** Build a thorough regex baseline covering all major patterns (§, Art., Abs., Anhang, Verordnung, Richtlinie, Tz., lit., Nr., S.). Make it genuinely competitive.
**Phase:** Evaluation

### 9. LLM Non-Determinism Breaking Reproducibility
**Risk:** Even with fixed seeds, LLM APIs don't guarantee deterministic output. Two training runs with "same seed" produce different data.
**Warning signs:** Metrics vary significantly between runs with same config.
**Prevention:** Cache generated data after first generation. Use cached data for reproducibility studies. Document that online generation is non-deterministic by nature.
**Phase:** Data generation

### 10. Max Sequence Truncation Cutting References
**Risk:** German legal references can be long ("Artikel 6 Absatz 1 Unterabsatz 1 Buchstabe a der Verordnung (EU) 2016/679"). With max_seq_length=256, truncation may cut a reference mid-span.
**Warning signs:** Truncated samples have partial BIO sequences (B-REF at end without closing).
**Prevention:** Handle truncation in BIO converter — if a reference span crosses the truncation boundary, either drop the partial span or extend the window. Log truncation frequency.
**Phase:** Data pipeline

### 11. fp16 NaN on MPS Backend
**Risk:** Mixed precision (fp16) is not fully stable on Apple MPS backend. Can produce NaN gradients.
**Warning signs:** NaN loss values during training on Mac.
**Prevention:** Auto-detect device: use fp16 only on CUDA, disable on MPS/CPU. Add NaN detection in training loop with gradient clipping.
**Phase:** Training implementation

## Minor Pitfalls

### 12. O-Class Imbalance
**Risk:** With ~40% texts having no references and references being short spans, the O label dominates (often >90% of tokens). Model can achieve high accuracy by predicting all-O.
**Warning signs:** Model loss decreases but F1 stays near 0; predictions are all-O.
**Prevention:** Use class weights in CrossEntropyLoss (upweight B-REF and I-REF). Monitor per-class metrics during training. The 60/40 reference ratio in generation helps but token-level imbalance persists.
**Phase:** Training implementation

### 13. German Unicode Normalization
**Risk:** German text from PDFs may contain different Unicode representations (composed vs decomposed umlauts, different dash types — em-dash, en-dash, minus). Tokenizer may handle these differently.
**Warning signs:** Character offsets don't match after tokenization; BIO alignment breaks on certain texts.
**Prevention:** Normalize all input text (NFC normalization). Standardize dash/quote characters. Test with edge cases (ä vs a+combining umlaut).
**Phase:** Data pipeline

### 14. Ensemble Caching Without Size Limits
**Risk:** Caching all generated training data for ensemble without size limits fills disk, especially with many epochs × large batch sizes.
**Warning signs:** Disk usage grows unbounded during ensemble training.
**Prevention:** Implement cache size limits. Use LRU or epoch-based cache eviction. Log cache size.
**Phase:** Ensemble implementation

---
*Researched: 2026-03-13*
