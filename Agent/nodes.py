import json
import traceback
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict
from Agent.agent_tools import EmbeddingDBUtils, local_search, web_search
from Agent.utils import parse_planner_response
from Agent.prompts import writer_prompt

# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = PROJECT_ROOT / "Bert_model" / "embedding_model"
RERANK_DIR = PROJECT_ROOT / "Bert_model" / "reranker_model"
LLM_DIR = PROJECT_ROOT / "LLM"
CHROMA_DIR = PROJECT_ROOT / "Chroma_db"


llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8888/v1",
    model_name=str(LLM_DIR / "Qwen3-8B"),
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    }
)
tools = [local_search, web_search]
prompt = writer_prompt

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
rag_utils = EmbeddingDBUtils()

# 定义 State
class RAGState(TypedDict):
    user_input: str
    query: str
    docs: list
    answer: str
    check_result: str
    continue_chat: str
    plan: str


# 用户输入节点
def get_user_input_node(state: RAGState) -> RAGState:
    user_input = input("\n👤 请输入你的问题（输入 exit 退出）：")
    if user_input.lower() in ["exit", "quit", "end", "滚"]:
        return {"continue_chat": "end"}
    return {"user_input": user_input,
            "continue_chat": "continue"}

# 节点：提取query
def extract_query_node(state: RAGState) -> RAGState:
    if "写综述：" in state["user_input"]:
        query = state["user_input"].strip()
    else:
        query = rag_utils.query_extract(state["user_input"])
    print(query)
    return {"query": query}


# 节点：文档检索
def document_retrieve_node(state: RAGState) -> RAGState:
    results = rag_utils.db_query(  # 使用全局工具类实例
        query=state["query"],
        top_k=3,
        rerank=True
    )
    return {"docs": results}

# 节点：生成答案
def generate_answer_node(state: RAGState) -> RAGState:
    retrieved_results=state["docs"]
    user_question=state["user_input"]
    answer = rag_utils.single_chat_think(user_question, retrieved_results)
    print(answer)
    return {"answer": answer}

# 节点：验证答案
def self_check_node(state: RAGState) -> RAGState:
    retrieved_results=state["docs"]
    answer=state["answer"]
    check = rag_utils.self_check(retrieved_results, answer)
    print("检查是否无幻觉......", check)
    return {"check_result": check}

# 判断节点
def check_continue_node(state: RAGState):
    user_input = input("\n👤 是否继续提问？(y/n): ")
    return {"continue_chat": "continue" if user_input.lower() == "y" else "end"}

# 综述plan节点
def planner_node(state: RAGState) -> RAGState:
    user_question=state["user_input"].replace("写综述：", "").strip()
    plan = rag_utils.review_planner(user_question)
    parse_plan = json.dumps(parse_planner_response(plan), ensure_ascii=False)
    return {"plan": parse_plan}

def _process_one_section(section, idx):
    """单个 section 的检索 + 生成，返回 {id,title,content}"""
    try:
        # 1) 构造检索 query（用 retrieval_hints 前 12 个拼接）
        hits_query = " ".join(section.get("retrieval_hints", [])[:12]) or section.get("title", "")

        # 2) RAG 检索（可按需调小 top_k 降时延）
        evidences = rag_utils.db_query(
            query=hits_query,
            top_k=5,
            rerank=True
        )

        # 3) 生成该段（按你当前接口：title + evidences；若你有 section_json/evidence_whitelist 的版本就替换这里）
        content = rag_utils.section_writer(section.get("title", ""), evidences)

        return {"id": idx, "title": section.get("title", f"section-{idx}"), "content": content}

    except Exception as e:
        # 出错也要占位，避免整篇卡死
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        return {
            "id": idx,
            "title": section.get("title", f"section-{idx}"),
            "content": f"文段生成失败：{err}"
        }

def section_writer_node(state: RAGState) -> RAGState:
    plan = json.loads(state["plan"])["sections"]
    max_workers = min(4, len(plan))

    results = [None] * len(plan)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="secw") as ex:
        fut2idx = {ex.submit(_process_one_section, section, idx): idx
                   for idx, section in enumerate(plan)}

        # 可选：超时控制（单段超时不影响其他段）
        for fut in as_completed(fut2idx, timeout=None):
            idx = fut2idx[fut]
            try:
                results[idx] = fut.result(timeout=None)
            except Exception as e:
                results[idx] = {
                    "id": idx,
                    "title": plan[idx].get("title", f"section-{idx}"),
                    "content": f"文段生成失败：{e}"
                }
    answers = sorted((r for r in results if r is not None), key=lambda x: x["id"])
    return {"answer": answers}

def section_agent_writer_node(state: RAGState) -> RAGState:
    plan = json.loads(state["plan"])["sections"]
    answers = []
    for section in plan:
        answer = agent_executor.invoke({"input": section})
        answers.append(answer)
    return {"answer": answers}


def assembler_answer_node(state: RAGState) -> RAGState:
    answers = state["answer"]
    answer = rag_utils.assembler_answer(answers)
    
    with open("answer.md", "w", encoding="utf-8") as f:
        f.write(answer)
    print("finish!")
    return {"answer": answer}
