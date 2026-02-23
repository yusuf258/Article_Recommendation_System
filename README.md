# Article Recommendation System | Content-Based Filtering

Content-based article recommendation engine using TF-IDF vectorization and cosine similarity to suggest related articles.

## Problem Statement
Given an article, automatically recommend the **most similar articles** from the corpus based on textual content similarity — without requiring any user history or ratings.

## Dataset
| Attribute | Detail |
|---|---|
| Articles | 34 articles with titles and content |
| Method | Content-based (no user interaction required) |
| Output | Top-4 most similar article recommendations |

## Methodology
1. **Data Loading** — Article titles and content ingestion
2. **Text Preprocessing** — Lowercasing, stopword removal, punctuation cleaning
3. **TF-IDF Vectorization** — Convert article text to numerical feature vectors
4. **Cosine Similarity Matrix** — Compute pairwise similarity between all articles
5. **Recommendation Function** — Return Top-N most similar articles for any given article
6. **Model Persistence** — Save vectorizer and similarity matrix for production use

## Results
| Component | Detail |
|---|---|
| Corpus Size | 34 articles |
| Similarity Metric | Cosine Similarity |
| Output | Top-4 recommendations per article |

> Cold-start friendly: no user data needed. Recommendations are based purely on article content.

## Technologies
`Python` · `scikit-learn` · `Pandas` · `NumPy` · `joblib`

## File Structure
```
06_Article_Recommendation_System/
├── project_notebook.ipynb   # Main notebook
├── articles.csv             # Article dataset
└── models/                  # Saved vectorizer and similarity matrix
```

## How to Run
```bash
cd 06_Article_Recommendation_System
jupyter notebook project_notebook.ipynb
```
