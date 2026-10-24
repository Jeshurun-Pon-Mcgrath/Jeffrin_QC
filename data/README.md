# Data card

registry.csv contains 72 invented English publication-style titles in 24 subject families. No title is asserted to be registered or legally available. These are synthetic examples, not official SIH or Press Registrar data. Similarity labels are educational annotations, not regulatory decisions.

The evaluation creates punctuation/case changes, missing articles, word reordering, and a small spelling edit as positives. Other formats in the same family and titles from another family are negatives. These rules approximate duplicate detection, not all semantic similarity. All pairs sharing a subject family are kept in one split; negative candidates also come from that split. No test labels are used for threshold selection. The vocabulary for retrieval is fitted on the registry, as in a search index; the pair classifier is trained only on training pairs.

Limitations: small template-based data, English only, narrow typo patterns, no authoritative legal labels, phonetic or transliterated matches, real multilingual titles, and semantic paraphrases. Excellent synthetic scores do not estimate production accuracy. For deployment collect a licensed real registry and independently annotated submissions; freeze an external test set before tuning. Obtain consent where needed and avoid personal information.
