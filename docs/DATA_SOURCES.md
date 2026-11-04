# Dataset Provenance & Data Sources

GroundUpGPT utilizes strictly public, legally licensed, and verifiable data sources. No scraped web crawl data or proprietary foundation model outputs are used.

## 1. Corpus Sources & Licensing

| Source | Domain / Purpose | License | Volume / Split |
|---|---|---|---|
| **Project Gutenberg** | Classic English literature, historical prose, philosophy | Public Domain (US) | 62 books (~13.87M train tokens, 1.12M held-out val tokens) |
| **Simple English Wikipedia** | Modern expository articles, definitions, facts | CC BY-SA 4.0 / GFDL | 57,013 clean articles (~21.77M train tokens, 1.13M val tokens) |
| **Databricks Dolly 15k** | Conversational instruction following and short answers | CC BY-SA 3.0 | 9,370 sanitized pairs (8,433 train, 937 held-out val) |

## 2. Preprocessing & Corpus Transformations

- **Normalization**: All texts undergo Unicode NFKC normalization, removing erratic control codes while preserving standard punctuation.
- **Boilerplate Stripping**: Project Gutenberg header/transcriber notices and Wikipedia navigational tables/infoboxes are programmatically stripped.
- **Document Boundary Delimiters**: Distinct articles and chapters are demarcated with explicit `<|endoftext|>` boundary tokens.
- **Zero-Leak Validation Shingling**: Training and validation splits are strictly partitioned by whole documents and verified with 10-word shingle index checks to prevent train-validation leakage.
- **Conversational Loss Masking**: Instruction datasets apply assistant-only loss masking, setting cross-entropy target labels for the `<|user|>` turn and delimiters to `-100` so gradients update only on model responses.

## 3. Token Occurrences vs. Vocabulary Entries

To avoid common machine learning reporting ambiguities:

- **Token Occurrences**: The running total of serialized token IDs in the dataset stream (e.g., 50,301,446 pretraining tokens in Phase 1; 22,894,572 token occurrences in the Wikipedia V2 expansion).
- **Unique Vocabulary Entries**: The discrete subword pieces defined in the tokenizer vocabulary (strictly **4,096** entries in `checkpoints/bpe_vocab4096.json`).
- **Token Exposures**: The number of input tokens presented to the model during training (optimizer steps $\times$ effective batch size $\times$ context length). Random slice sampling across pretraining draws windows with replacement, meaning ~0.97 equivalent passes does not imply 100% unique token coverage.
