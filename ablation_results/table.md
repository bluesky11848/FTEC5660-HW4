| Variant | Change | Recall@5 | Kept | Rejected | Accuracy | Retrieval s |
|---|---|---:|---:|---:|---:|---:|
| B0 | Naive get_text + recursive split + whitespace BM25 | 28.3% | 100.0% | 0.0% | 50.0% | 0.137 |
| V1 | Replace HTML tables with labelled table text | 28.3% | 100.0% | 0.0% | 45.0% | 0.275 |
| V2 | Use section-aware chunks with section/caption headers | 28.3% | 100.0% | 0.0% | 41.7% | 0.311 |
| V3 | Use regex tokens, stop words and lexical aliases | 19.6% | 100.0% | 0.0% | 36.7% | 0.298 |
| V4 | Parse queries and filter company/year/document metadata | 63.0% | 100.0% | 71.4% | 73.3% | 0.392 |
| V5 | Remove company/year terms from the ranking query | 100.0% | 100.0% | 71.4% | 100.0% | 0.38 |
| V6 | Allocate comparison evidence per company | 100.0% | 100.0% | 71.4% | 100.0% | 0.386 |
| V7 | Reject unknown companies, unsupported years/metrics | 100.0% | 100.0% | 100.0% | 100.0% | 0.425 |
