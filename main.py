from Agent.rag_tools import EmbeddingDBUtils
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
LLM_DIR = PROJECT_ROOT / "LLM"
EMBEDDING_DIR = PROJECT_ROOT / "Bert_model" / "embedding_model"
RERANK_DIR = PROJECT_ROOT / "Bert_model" / "reranker_model"
CHROMA_DIR = PROJECT_ROOT / "Chroma_db"
EmbeddingDBUtils.configure_llm(
            api_key="EMPTY",
            api_base="http://localhost:8888/v1",
            model_name=str(LLM_DIR / "Qwen3-8B"),
            embedder_name=str(EMBEDDING_DIR / "bge-m3-ft"),
            reranker_name=str(RERANK_DIR / "bge-reranker-v2-m3-ft"),
            db_name=str(CHROMA_DIR / "bge-m3-ft")
        )
rag_utils = EmbeddingDBUtils()
from Agent.workflow import build_workflow


def main():
    # 执行工作流
    workflow = build_workflow()
    result = workflow.invoke({})


if __name__ == "__main__":
    main()
