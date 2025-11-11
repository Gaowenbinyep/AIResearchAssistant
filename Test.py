from openai import OpenAI, AsyncOpenAI
from pathlib import Path
from datasets import Dataset
import sys
import re
import os
from tqdm import tqdm
import pandas as pd
from langchain_chroma import Chroma
from FlagEmbedding import BGEM3FlagModel
from langchain.embeddings.base import Embeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, AnswerAccuracy, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import asyncio  # 需要添加此行导入

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent
data_dir = project_root / "Data"
for path in (data_dir, project_root):
    if str(path) not in sys.path:
        sys.path.append(str(path))
from Agent.agent_tools import LangChainBGE, EmbeddingDBUtils

llm_dir = project_root / "LLM"
embedding_dir = project_root / "Bert_model" / "embedding_model"
rerank_dir = project_root / "Bert_model" / "rerank_model"
chroma_dir = project_root / "Chroma_db"

# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
model_name = str(llm_dir / "RAG_1.7B_sft_v2")
embedder_name = str(embedding_dir / "bge-m3-ft")
reranker_name = str(rerank_dir / "bge-reranker-v2-m3-ft")
db_name = str(chroma_dir / "v1.0" / "bge-m3-ft")

# 云端测试模型
os.environ["DASHSCOPE_API_KEY"] = "sk-****************************"
MODEL = "qwen-plus"

def get_score(doc_id, recall_ids, top_k):
    if doc_id in recall_ids:
        return (top_k - recall_ids.index(doc_id))/top_k
    return 0

def get_evaluation(questions, answers, contexts, ground_truths):
    
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset=dataset,
        llm=ChatOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=MODEL,
            temperature=0.0,
            top_p=0.9,
            extra_body={"enable_thinking": False}
        ),
        embeddings=LangChainBGE(
            model_name=str(embedding_dir / "bge-m3"),
            use_fp16=True
        ),
        metrics=[
            faithfulness,
            answer_relevancy,
            AnswerAccuracy()
            # context_precision
        ]
    )
    df = result.to_pandas()
    os.makedirs("./test_results", exist_ok=True)
    df.to_csv("./test_results/evaluation_details.csv", index=False, encoding="utf-8")

    metrics_avg = df[["faithfulness", "answer_relevancy", "nv_accuracy"]].mean()
    
    return metrics_avg

# 需要调整导入语句（文件开头）
from tqdm import tqdm as sync_tqdm  # 保留同步tqdm供其他代码使用
from tqdm.asyncio import tqdm as async_tqdm  # 添加异步tqdm

async def rag_work(test_data_path, think=False):
    test_datas = pd.read_json(test_data_path, lines=True)
    query_pattern = re.compile(r'<query>(.*?)<\/query>', re.DOTALL)
    answer_pattern = re.compile(r'<answer>(.*?)<\/answer>', re.DOTALL)
    context_pattern = re.compile(r'<参考文本>(.*?)<\/参考文本>', re.DOTALL)
    conversations = test_datas["conversations"].tolist()
    rag_utils = EmbeddingDBUtils()
    
    # 创建信号量控制并发数量（根据系统性能调整）
    semaphore = asyncio.Semaphore(10)
    
    # 异步处理单个对话的函数
    async def process_conversation(conversation):
        async with semaphore:  # 限制并发数量
            question = query_pattern.search(conversation[1]["content"])
            ground_truth = answer_pattern.search(conversation[2]["content"])
            context = context_pattern.search(conversation[1]["content"])
            
            if not (question and ground_truth and context):
                return None
                
            question = question.group(1)
            context = context.group(1)
            ground_truth = ground_truth.group(1)
            answer = None
            
            if think:
                # 异步调用同步方法并添加重试逻辑
                for try_time in range(11):  # 最多重试10次
                    # 使用to_thread包装同步模型调用
                    answer_text = await asyncio.to_thread(
                        rag_utils.single_chat_think, question, context
                    )
                    answer = answer_pattern.search(answer_text)
                    if answer:
                        answer = answer.group(1)
                        break
            else:
                # 异步调用普通对话方法
                answer = await asyncio.to_thread(rag_utils.single_chat, question, context)
                
            return (question, answer, context, ground_truth) if answer else None
    
    # 创建所有异步任务
    tasks = [process_conversation(conv) for conv in conversations]
    
    # 并发执行所有任务并显示进度条
    results = []
    for future in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理对话数据", unit="条", ncols=100):
        result = await future
        results.append(result)
    
    # 收集结果（保持原有逻辑）
    questions, answers, contexts, ground_truths = [], [], [], []
    for res in results:
        if res:
            q, a, c, gt = res
            questions.append(q)
            answers.append(a)
            contexts.append([c])
            ground_truths.append(gt)
    
    metrics_avg = get_evaluation(questions, answers, contexts, ground_truths)
    return metrics_avg



if __name__ == "__main__":
     # 配置LLM参数（全局仅需调用一次）
    EmbeddingDBUtils.configure_llm(
        api_key=openai_api_key,
        api_base=openai_api_base,
        model_name=model_name,
        embedder_name=embedder_name,
        reranker_name=reranker_name,
        db_name=db_name
    )
    top_k = 3
    
    # # 获取工具类实例（自动初始化Embedding/向量库/LLM）
    # rag_utils = EmbeddingDBUtils()
    # datas = pd.read_json(path_or_buf="/media/a822/82403B14403B0E83/Gwb/RAG/Test/QApairs_query_abs.jsonl", lines=True)
    # outputs = []
    # for _, data in datas.iterrows():
    #     query = data["key_words"]
    #     doc_id = data["id"]
    #     docs = rag_utils.db_query(query, top_k=top_k, rerank=True)
    #     recall_ids = [doc.id for doc in docs]
    #     score = get_score(doc_id=doc_id, recall_ids=recall_ids, top_k=top_k)
    #     outputs.append({
    #         "query": query,
    #         "doc_id": doc_id,
    #         "recall_ids": recall_ids,
    #         "score": score
    #     })
    # pd.DataFrame(outputs).to_json("./test_results/rag_output.jsonl", orient="records", lines=True, force_ascii=False)
    # top_k_acc = sum([1 if o["score"] > 0 else 0 for o in outputs])/len(outputs)
    # Map = sum([o["score"] for o in outputs])/len(outputs)
    # # 计算top_k准确率和Map
    # print(f"Top-{top_k}准确率: {top_k_acc:.4f}")
    # print(f"Map: {Map:.4f}")
    

    metrics_avg = asyncio.run(rag_work(
        test_data_path=str(project_root / "Data" / "Train" / "llm_test_data.jsonl"), 
        think=True
    ))
    print(metrics_avg)
