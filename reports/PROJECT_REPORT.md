# TitleGuard: project report and demonstration guide

## Abstract
TitleGuard screens a proposed publication title against a small registry and returns similar existing titles for human review. It uses content-based retrieval with word and character TF-IDF, then a logistic regression pair classifier. The CPU notebook includes data analysis, statistical exploration, clustering, evaluation and an interactive submission form.

## Problem and objectives
Manual title checking can overlook spelling, punctuation or ordering variations. The objective is to surface potential conflicts with interpretable candidates, measure retrieval and pair classification separately, and communicate uncertainty. The prototype is inspired by SIH1782; it does not implement official regulatory criteria or connect to an authoritative registry.

## Methodology
1. Import 72 synthetic titles, inspect missing values and preserve provenance.
2. Normalize Unicode, case and punctuation while retaining words and numbers. This can overmerge punctuation-sensitive names and requires review.
3. Index the available registry with word bigrams and character 2–5 grams.
4. Rank candidates using a 40:60 blend of word and character cosine scores.
5. Train logistic regression on five symmetric pair features using only training families.
6. Tune a review threshold on validation pairs, then freeze it for held-out testing.
7. Flag exact normalized matches explicitly and show all top-five candidates for review.

The classifier uses educational rule-generated examples, not human legal judgments. Its scores are not calibrated probabilities. The benchmark has 252 training, 90 validation and 90 test pairs. Real deployment requires licensed registry data, independent labelers, a documented similarity policy, larger external testing, and language-specific evaluation.

## Results and interpretation
Held-out accuracy is 93.33%, precision 100%, recall 90%, and F1 94.74%. There are 30 true negatives, 54 true positives, zero false positives, and six false negatives. Retrieval returns the intended title in the first five candidates for all 60 positive test queries. The fixed Jaccard baseline F1 is 66.67%. Results apply only to the generated benchmark. Five test families provide limited uncertainty estimation.

The exploratory Welch procedure checks for degenerate variance before testing. In this template dataset, family mean Jaccard values have no within-class variation, so a meaningful inferential p-value cannot be estimated. The notebook reports this limitation explicitly. K-means exploration has a low silhouette; clusters are not proof of legal title categories.

## Demonstration sequence
Run every cell, explain the dataset and family split, inspect the confusion matrix and six errors, then enter `THE RIVER CHRONICLE!` for an exact normalized match and `River Chroncile` for a spelling variation. Enter an unrelated title to show that low similarity does not mean approval. Submit punctuation-only input to show validation. Show the top candidates and explain the two score columns.

## Responsible use and future work
Do not store confidential titles in a shared notebook. Evaluate multilingual and transliterated names, phonetic matching, semantic embeddings, real title policies, calibration, and reviewer agreement on licensed data. A production service would also need persistent records, access control, audit trails and operational monitoring. Do not use this prototype to make final legal decisions.

## References
- scikit-learn User Guide: text feature extraction, logistic regression, clustering and model evaluation: https://scikit-learn.org/stable/user_guide.html
- SciPy Welch t-test documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- SIH1782 problem description supplied by the student; no additional official specifications or registry data were supplied.
