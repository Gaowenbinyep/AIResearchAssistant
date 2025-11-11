from pathlib import Path

from langgraph.graph import StateGraph, END

from Agent.agent_tools import EmbeddingDBUtils
from .nodes import RAGState, get_user_input_node, extract_query_node, document_retrieve_node, generate_answer_node, check_continue_node, self_check_node
from .nodes import section_writer_node, planner_node, assembler_answer_node, section_agent_writer_node

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_workflow():
    # 创建工作流图
    graph = StateGraph(RAGState)
    
    graph.add_node("get_user_input", get_user_input_node)
    graph.add_node("extract_query", extract_query_node)
    graph.add_node("document_retrieve", document_retrieve_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("self_check", self_check_node)
    graph.add_node("check_continue", check_continue_node)
    
    graph.add_node("planner", planner_node)
    graph.add_node("section_writer", section_agent_writer_node)
    graph.add_node("assembler_answer", assembler_answer_node)



    graph.set_entry_point("get_user_input")
    graph.add_edge("document_retrieve", "generate_answer")
    graph.add_edge("generate_answer", "self_check")
    graph.add_edge("planner", "section_writer")
    graph.add_edge("section_writer", "assembler_answer")



    graph.add_conditional_edges(
        "extract_query",
        lambda state: "planner" if state["user_input"].startswith("写综述：") else "retrieve",
        {
            "planner": "planner",
            "retrieve": "document_retrieve"
        }
    )
    graph.add_conditional_edges(
        "self_check",
        lambda state: state["check_result"],
        {
            "true": "check_continue",
            "false": "generate_answer"
        }
    )

    graph.add_conditional_edges(
        "check_continue",
        lambda state: state["continue_chat"],
        {
            "continue": "get_user_input",
            "end": END
        }
    )
    graph.add_conditional_edges(
        "get_user_input",
        lambda state: state["continue_chat"],
        {
            "continue": "extract_query",
            "end": END
        }
    )
    # 构建工作流
    workflow = graph.compile()

    return workflow


if __name__ == "__main__":

    EmbeddingDBUtils.configure_llm(
        api_key="EMPTY",
        api_base="http://localhost:8888/v1",
        model_name=str(PROJECT_ROOT / "LLM" / "RAG_1.7B_sft_v2"),
        embedder_name=str(PROJECT_ROOT / "Bert_model" / "embedding_model" / "bge-m3-ft"),
        reranker_name=str(PROJECT_ROOT / "Bert_model" / "reranker_model" / "bge-reranker-v2-m3-ft"),
        db_name=str(PROJECT_ROOT / "Chroma_db" / "bge-m3-ft")
    )
    # 执行工作流
    workflow = build_workflow()
    
    result = workflow.invoke({})
