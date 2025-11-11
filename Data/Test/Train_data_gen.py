import pandas as pd
import random
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from langchain_chroma import Chroma
from FlagEmbedding import BGEM3FlagModel
from langchain.embeddings.base import Embeddings
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
test_dir = project_root / "Data" / "Test"
train_dir = project_root / "Train"


def generate_train_data(QApairs_path):
    datas = pd.read_json(QApairs_path, lines=True)
    new_datas = []
    all_contents = datas['content'].tolist()
    for _, data in tqdm(datas.iterrows(), total=len(datas)):
        key_words = data["key_words"]
        pos_contents = data["content"]
        neg_contents = random.choices(all_contents, k=9)
        while pos_contents in neg_contents:
            neg_contents = random.choices(all_contents, k=9)
        new_datas.append({
            "query": key_words,
            "pos": [pos_contents],
            "neg": neg_contents,
            "pos_scores": [1.0],
            "neg_scores": [0.0] * len(neg_contents),
            "prompt": "给定查询和文档，判断查询和文档的内容是否相关",
            "pos_id": data["id"]
        })
    pd.DataFrame(new_datas).to_json(
        train_dir / "train_data.jsonl", lines=True, orient="records", force_ascii=False
    )

if __name__ == "__main__":
    generate_train_data(test_dir / "QApairs_query_abs.jsonl")
