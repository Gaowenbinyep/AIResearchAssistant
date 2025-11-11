import pandas as pd
import os
import random
import json
import re
from tqdm import tqdm
from openai import OpenAI
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
data_dir = current_file_path.parent.parent
test_dir = current_file_path.parent
project_root = current_file_path.parents[2]
for path in (data_dir, project_root):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from Agent.agent_tools import EmbeddingDBUtils


os.environ["DASHSCOPE_API_KEY"] = "sk-****************************"

# ======================================
# 配置区
MODEL = "qwen-plus"  # 替换成可用的模型名
OUTPUT_FILE = str(test_dir / "QApairs.jsonl")
# ======================================


def call_qwen(prompt):
    """ 调用 Qwen3-235B 返回结果 """
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=prompt,
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump_json()
    score = json.loads(response)["choices"][0]["message"]["content"]
    return score


def build_prompt(query):
    prompt = [{
                "role": "system",
                "content": f"""你是外骨骼控制领域的文献检索专家，专为实验室研究人员设计检索query。
                    任务：根据给定的外骨骼控制领域文献文本，生成3个专业、精准的检索query，并给出文章的作者，用于RAG系统查找相关学术文献。
                    要求：
                    1. 必须基于文本核心内容，聚焦外骨骼控制技术关键点（如控制算法、人机交互、动力学建模、运动意图识别等）
                    2. 使用领域专业术语（如"外骨骼机器人""肌电信号控制""阻抗控制""协同运动""穿戴舒适性"等）
                    3. 符合科研人员检索习惯，可包含组合关键词（如"外骨骼机器人 自适应控制 中风康复"）
                    4. 避免过于宽泛（如仅"外骨骼"）或过于狭窄（如包含具体实验数据）的表述
                    5. 每条query独立聚焦文本不同维度，形成检索策略组合
                    """
                },
            {
                "role": "user",
                "content": f"""
                <文献文本>
                {query}
                </文献文本>

                请输出3条检索query，作者，按如下格式返回，示例：
                <query>
                    髋关节外骨骼 自适应阻抗控制
                    肌电信号 运动意图识别
                    IMU传感器 膝关节外骨骼控制
                </query>
                <author>
                    所有作者的名称
                </author>
                请严格按照上述格式输出，不允许输出其他内容。
                """
            }]
    return prompt

def format_data(QApairs_path, output_path):
    datas = pd.read_json(QApairs_path, lines=True)
    new_datas = []
    for _, data in tqdm(datas.iterrows(), total=len(datas)):
        querys = data["query"]
        authors = data["author"].split(", ")
        content = data["content"]
        id = data["id"]
        metadata = data["metadata"]
        for query in querys:
            new_datas.append({
                "query": query,
                "author": random.choice(authors),
                "content": content,
                "id": id,
                "metadata": metadata
            })
    random.shuffle(new_datas)
    pd.DataFrame(new_datas).to_json(output_path, lines=True, orient="records", force_ascii=False)

async def async_call_qwen(prompt):
    """异步调用Qwen API"""
    client = AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=prompt,
            extra_body={"enable_thinking": False},
        )
        response = completion.model_dump_json()
        return json.loads(response)["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API调用错误: {str(e)}")
        return None

# 异步处理单条数据
async def process_single_data(data, semaphore):
    async with semaphore:  # 控制并发数量
            
        prompt = build_answer_prompt(data["query"], data["content"])
        res = None
        
        # 异步重试机制
        for _ in range(10):
            try:
                res = await async_call_qwen(prompt)
                if res:
                    break
            except openai.BadRequestError as e:
                print(f"Skipping problematic document {data['id']}: {str(e)}")
                return None
                
        if not res:
            return None
            
        # 解析结果
        pattern = re.compile(r'<response>(.*?)</response>', re.DOTALL)
        matches = pattern.findall(res)
        if not matches:
            return None
            
        response = matches[0].strip()
        print(response)
        
        return {
            "query": data["query"],
            "response": response,
            "content": data["content"],
            "id": data["id"],
            "metadata": data["metadata"]
        }

# 主异步函数
async def async_process_data():
    # 读取数据
    datas = pd.read_json(
        str(test_dir / "QApairs_query_content.jsonl"),
        lines=True
    )
    
    data_list = datas.to_dict("records")
    semaphore = asyncio.Semaphore(5)
    
    tasks = [
        process_single_data(data, semaphore) 
        for data in random.choices(data_list, k=800)
    ]
    
    results = await tqdm_asyncio.gather(
        *tasks, 
        desc="处理进度"
    )
    
    lines = [result for result in results if result is not None]
    pd.DataFrame(lines).to_json(
        str(test_dir / "QApairs_query_answer_content.jsonl"),
        orient="records", 
        force_ascii=False, 
        lines=True
    )




if __name__ == "__main__":

    # EmbeddingDBUtils.configure_llm(
    #         api_key="EMPTY",
    #         api_base="http://localhost:8888/v1",
    #         model_name="/media/a822/82403B14403B0E83/Gwb/WechatRobot/Base_models/Qwen3-1.7B",
    #         embedder_name="/media/a822/82403B14403B0E83/Gwb/RAG/embedding_model/bge-m3-ft",
    #         reranker_name = "/media/a822/82403B14403B0E83/Gwb/RAG/Rerank_model/bge-reranker-v2-m3_ft",
    #         db_name="/media/a822/82403B14403B0E83/Gwb/RAG/chroma_db/v1.0/bge-m3-ft"
    #     )
    # rag_utils = EmbeddingDBUtils()
    # datas = rag_utils.db_iterate(filter={"type": "abstract"})
    # lines = []
    # for data in tqdm(datas):
    #     prompt = build_prompt(data["page_content"])
    #     if len(data["page_content"]) < 300:
    #         continue
    #     res = call_qwen(prompt)
    #     pattern = re.compile(r'<query>(.*?)</query>', re.DOTALL)
    #     pattern_author = re.compile(r'<author>(.*?)</author>', re.DOTALL)
    #     matches = pattern.findall(res)
    #     author = pattern_author.findall(res)
    #     if not matches:
    #         for i in range(10):
    #             res = call_qwen(prompt)
    #             pattern = re.compile(r'<query>(.*?)</query>', re.DOTALL)
    #             matches = pattern.findall(res)
    #             if matches:
    #                 break
    #     if matches:
    #         if not author:
    #             author = "None"
    #         query = matches[0].strip().split("\n")
    #         query = [q.strip() for q in query]
    #         author = author[0].strip()
    #         lines.append({
    #             "query": query,
    #             "author": author,
    #             "content": data["page_content"],
    #             "id": data["id"],
    #             "metadata": data["metadata"]
    #         })
    # pd.DataFrame(lines).to_json(OUTPUT_FILE, orient="records", force_ascii=False)

    format_data(OUTPUT_FILE, "./QApairs_lines.jsonl")
