# Retrieval evaluation

3Blue1Brown - But what is a neural network? | Deep learning chapter 1  
21 questions, embeddings: `text-embedding-3-small`

| Chunk size | Chunks | k | Hit rate | MRR | Video retrieved |
|---:|---:|---:|---:|---:|---:|
| 500 | 55 | 1 | 62% | 0.62 | 3% |
| 500 | 55 | 3 | 81% | 0.71 | 7% |
| 500 | 55 | 5 | 90% | 0.74 | 12% |
| 1000 | 22 | 1 | 90% | 0.90 | 5% |
| 1000 | 22 | 3 | 90% | 0.90 | 15% |
| 1000 | 22 | 5 | 100% | 0.93 | 25% |
| 1500 | 14 | 1 | 86% | 0.86 | 8% |
| 1500 | 14 | 3 | 95% | 0.90 | 23% |
| 1500 | 14 | 5 | 100% | 0.91 | 38% |
| 2000 | 10 | 1 | 76% | 0.76 | 11% |
| 2000 | 10 | 3 | 90% | 0.83 | 31% |
| 2000 | 10 | 5 | 95% | 0.84 | 51% |
