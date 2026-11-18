| Variant | Change | Recall@5 | Kept | Rejected | Accuracy | Retrieval s |
|---|---|---:|---:|---:|---:|---:|
| B0 | Naive get_text + recursive split + whitespace BM25 | 28.3% | 100.0% | 0.0% | 48.3% | 0.131 |
| V1 | Replace HTML tables with labelled table text | 28.3% | 100.0% | 0.0% | 45.0% | 0.271 |
| V2 | Use section-aware chunks with section/caption headers | 28.3% | 100.0% | 0.0% | 41.7% | 0.309 |
| V3 | Use regex tokens, stop words and lexical aliases | 19.6% | 100.0% | 0.0% | 38.3% | 0.291 |
| V4 | Parse and filter metadata; rank the remaining metric query | 100.0% | 100.0% | 71.4% | 100.0% | 0.377 |
| V5 | Allocate comparison evidence per company | 100.0% | 100.0% | 71.4% | 100.0% | 0.391 |
| V6 | Reject unknown companies, unsupported years/metrics | 100.0% | 100.0% | 100.0% | 100.0% | 0.416 |
