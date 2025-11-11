
from openai import OpenAI
from langchain.tools import tool
from pydantic import BaseModel, Field
from Agent.rag_tools import EmbeddingDBUtils, LangChainBGE

__all__ = [
    "EmbeddingDBUtils",
    "LangChainBGE",
    "local_search",
    "web_search",
]

rag_utils = EmbeddingDBUtils()


class LocalSearch(BaseModel):
    query: str = Field(description="要查询的信息")
@tool("local_search", args_schema=LocalSearch)
def local_search(query: str) -> str:
    """
        本地搜索工具，用于查询本地知识库。
        
        参数：
        query (str): 要查询的信息
        
        返回：
        str: 本地知识库中相关的信息
    """
    query = rag_utils.query_extract(query)
    results = rag_utils.db_query(
        query=query,
        top_k=3,
        rerank=True
    )
    return results


class WebSearch(BaseModel):
    query: str = Field(description="要查询的信息")
@tool("web_search", args_schema=WebSearch)
def web_search(query: str) -> str:
    """
        网络搜索工具，用于查询网络信息。
        
        参数：
        query (str): 要查询的信息
        
        返回：
        str: 网络中相关的信息
    """
    client = OpenAI(
        api_key="bce-v3/****************************",
        base_url="https://qianfan.baidubce.com/v2/ai_search"
    )
    response = client.chat.completions.create(
    model="deepseek-r1",
    messages=[
        {"role": "user", "content": "今天有哪些体育新闻"}
    ],
    stream=False
)
    print(response)
    return response.choices[0].message.content

if __name__ == "__main__":
    print(web_search("蒋介石的父亲叫什么"))
