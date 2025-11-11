import pandas as pd
import random
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import openai
import os
import json
os.environ["DASHSCOPE_API_KEY"] = "sk-****************************"
MODEL = "qwen-plus"

def build_reverse_query_prompt(key_words, author):
    """生成反向查询的prompt模板"""
    system_prompt = """
    你是一个专业的查询转换助手，需要将给定的学术关键词和作者信息转换为自然、口语化的中文问题。
    转换规则：
    1. 如果提供了作者（非"None"或空值），问题中必须包含作者姓名，格式如"[作者]关于[关键词组合]做了什么研究？"
    2. 如果未提供作者，直接使用关键词组合生成问题，格式如"给我找有关[关键词组合]的研究"
    3. 关键词需自然组合，避免生硬堆砌，符合日常口语表达习惯
    4. 问题长度控制在20-40字，简洁明了
    """
    
    user_prompt = f"""
    关键词：{key_words}
    作者：{author if author and author != "None" else "无"}
    
    请严格按照规则生成1条口语化查询问题，不可以更改关键词，无需额外解释。
    示例：
    关键词：外骨骼机器人 步态增强 自适应控制
    作者：Younbaek Lee
    
    回复示例：
    Younbaek Lee的有关外骨骼机器人的步态增强和自适应控制的相关研究是什么？
    给我找Younbaek Lee有关外骨骼机器人步态增强和自适应控制的研究论文。
    有哪些研究论文是关于Younbaek Lee的外骨骼机器人步态增强和自适应控制的？
    """
    
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

def build_reverse_query_prompt(key_words, author):
    """生成反向查询的prompt模板"""
    system_prompt = """
    你是一个专业的查询转换助手，需要将给定的学术关键词和作者信息转换为自然、口语化的中文问题。
    转换规则：
    1. 如果提供了作者（非"None"或空值），问题中必须包含作者姓名，格式如"[作者]关于[关键词组合]做了什么研究？"
    2. 如果未提供作者，直接使用关键词组合生成问题，格式如"给我找有关[关键词组合]的研究"
    3. 关键词需自然组合，避免生硬堆砌，符合日常口语表达习惯
    4. 问题长度控制在20-40字，简洁明了
    """
    
    user_prompt = f"""
    关键词：{key_words}
    作者：{author if author and author != "None" else "无"}
    
    请严格按照规则生成1条口语化查询问题，不可以更改关键词，无需额外解释。
    示例：
    关键词：外骨骼机器人 步态增强 自适应控制
    作者：Younbaek Lee
    
    回复示例：
    Younbaek Lee的有关外骨骼机器人的步态增强和自适应控制的相关研究是什么？
    给我找Younbaek Lee有关外骨骼机器人步态增强和自适应控制的研究论文。
    有哪些研究论文是关于Younbaek Lee的外骨骼机器人步态增强和自适应控制的？
    """
    
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

def call_qwen(prompt):
    """ 调用 Qwen3-325B 返回结果 """
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=prompt,
        temperature=1.0,
        top_p=0.9,
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump_json()
    response = json.loads(response)["choices"][0]["message"]["content"]
    return response

def generate_reverse_query(QApairs_path):
    datas = pd.read_json(QApairs_path, lines=True)
    new_datas = []
    for _, data in tqdm(datas.iterrows(), total=len(datas)):
        key_words = data["query"]
        author = data["author"]

        prompt = build_reverse_query_prompt(key_words, author)
        try:  # Add error handling
            response = call_qwen(prompt)
        except openai.BadRequestError as e:
            print(f"Skipping problematic document {data['id']}: {str(e)}")
            continue
        new_datas.append({
            "query": response,
            "key_words": author + " " + key_words,
            "content": data["content"],
            "id": data["id"]
        })
    pd.DataFrame(new_datas).to_json("./QApairs_query_abs.jsonl", lines=True, orient="records", force_ascii=False)



if __name__ == "__main__":
    # generate_reverse_query("./QApairs_lines_abs.jsonl")
