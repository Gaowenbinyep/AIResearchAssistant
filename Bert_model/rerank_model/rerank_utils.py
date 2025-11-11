from FlagEmbedding import FlagReranker



def rerank(query, passages):
    reranker = FlagReranker('./bge-reranker-v2-m3', use_fp16=True)
    score = reranker.compute_score([query, passages], normalize=True)
    return score
reranker = FlagReranker('./bge-reranker-v2-m3', use_fp16=True)
score = reranker.compute_score(['query', 'passage'], normalize=True)