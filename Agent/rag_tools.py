import os
import re
import pandas as pd
import hashlib
import random
import json
from tqdm import tqdm
from openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
import random
from langchain_chroma import Chroma
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from langchain.embeddings.base import Embeddings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = PROJECT_ROOT / "Bert_model" / "embedding_model"
RERANK_DIR = PROJECT_ROOT / "Bert_model" / "reranker_model"
LLM_DIR = PROJECT_ROOT / "LLM"
CHROMA_DIR = PROJECT_ROOT / "Chroma_db"




class LangChainBGE(Embeddings):
    def __init__(self, model_name='BAAI/bge-m3', device='cpu', use_fp16=True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, query_max_length=8192, show_progress=False)
        self.device = device

    def embed_documents(self, documents):
        outputs = self.model.encode(documents, batch_size=12, max_length=8192)
        dense_vecs = outputs['dense_vecs']
        return dense_vecs.tolist() if hasattr(dense_vecs, 'tolist') else dense_vecs

    def embed_query(self, query):
        outputs = self.model.encode([query], batch_size=1, max_length=8192)
        dense_vec = outputs['dense_vecs'][0]
        return dense_vec.tolist() if hasattr(dense_vec, 'tolist') else dense_vec

def document_extract(file_path):
    loader = UnstructuredPDFLoader(file_path, strategy="fast")
    documents = loader.load()
    for doc in documents:
        doc.page_content = re.sub(r'\(cid:\d+\)', '', doc.page_content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2600,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", "."]
    )

    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("\n", " ")
    return chunks

def annotate_entities(chunks, ner_pipe, score_threshold=0.7):
    for chunk in chunks:
        text = chunk.page_content
        ner_results = ner_pipe(text)

        # 过滤低置信度实体
        entities = []
        for res in ner_results:
            if res["score"] >= score_threshold:
                entities.append({
                    "entity": res["word"],         # 实体文本
                    "type": res["entity_group"],   # 实体类别 (Method, Task, Metric, Material...)
                    "score": float(res["score"])
                })
        
        chunk.metadata["entities"] = entities
    return chunks

def db_build(folder_path):
    embedding = LangChainBGE(
        model_name=str(EMBEDDING_DIR / "bge-m3"),
        use_fp16=True
    )
    db = Chroma(
        embedding_function=embedding,
        persist_directory=str(CHROMA_DIR / "v1.0" / "bge-m3")
    )
    
    all_valid_docs = []
    seen_contents = set()

    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            file_path = os.path.join(folder_path, file)
            documents = document_extract(file_path)
            valid_docs = [doc for doc in documents if doc.page_content.strip()]
            
            for i, doc in enumerate(valid_docs):
                doc.metadata['source'] = file
                doc.metadata['page_id'] = i + 1
                doc.metadata['type'] = 'abstract' if i == 0 else 'content'
                
                content_hash = hashlib.md5(doc.page_content.strip().encode()).hexdigest()
                if content_hash in seen_contents:
                    print(f"⚠️ 发现重复文档块：{file} (page {i+1})，已跳过")
                    continue
                seen_contents.add(content_hash)
                all_valid_docs.append(doc)

            if valid_docs:
                print(f"已缓存 {file} 的 {len(valid_docs)} 个非空文档块")
            else:
                print(f"⚠️ {file} 无有效文档块，已跳过")

    # 添加到 Chroma 向量库
    if all_valid_docs:
        batch_size = 1000 
        total_docs = len(all_valid_docs)
        for i in range(0, total_docs, batch_size):
            batch = all_valid_docs[i:i + batch_size]
            db.add_documents(batch)
            print(f"已添加 {min(i + batch_size, total_docs)}/{total_docs} 个文档块")
        
        print(f"向量数据库构建完成，共添加 {total_docs} 个文档块")
    else:
        print("向量数据库构建完成，无有效文档块")

def db_reencode(existing_db_path, new_embedding_model_path):
    """
    用新 embedding 模型重新对 Chroma 中的文档进行编码（不修改 page_content 和 metadata，只更新 embedding）
    """
    new_embedding = LangChainBGE(
        model_name=new_embedding_model_path,
        use_fp16=True
    )

    db = Chroma(
        embedding_function=new_embedding,
        persist_directory=existing_db_path
    )

    all_docs = db.get()
    if not all_docs["ids"]:
        print("⚠️ 向量库为空，无需重新编码")
        return

    doc_ids = all_docs["ids"]
    updated_docs = [
        Document(
            page_content=text,
            metadata=meta
        ) for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    
    db.update_documents(
        ids=doc_ids,
        documents=updated_docs
    )
    print(f"✅ 向量库重新编码完成，共更新 {len(updated_docs)} 个文档")


class EmbeddingDBUtils:
    """Embedding模型、向量库与LLM工具类（单例模式）"""
    _instance = None
    _embedding = None
    _db = None
    _llm_client = None
    _model_name = None

    _prompt_templates = {
        "single_chat": {
            "system": """
                你是外骨骼控制领域的专业论文回答专家，能基于文献文本生成可验证的深度解答。
                    任务：根据给定的研究query和参考文本，生成1个专业、系统的论文回答。
                    要求：
                    1. 若文献中无直接相关内容，需明确标注"文献无相关内容"，不得虚构信息
                    2. 尽可能使用领域专业术语回答。
            """,
            "query": """
                <query>
                {query}
                </query>
                
                <参考文本>
                {document}
                </参考文本>

                请输出1条专业的论文回答。
            """
        },
        "rag_prompt_think": {
            "system": """
                你是外骨骼控制领域的专业论文回答专家，具备分步推理能力，能基于文献文本生成可验证的深度解答。
                    任务：根据给定的研究query和参考文本，通过分步推理生成1个专业、系统的论文回答，并确保每步推理都有明确的文献依据。
                    要求：
                    1. 采用"分步推理法"：将复杂问题拆解为多个推理步骤，每个步骤聚焦一个子问题
                    2. 若文献中无直接相关内容，需明确标注"文献无相关内容"，不得虚构信息或自行推断文献内容外的信息
                    3. 使用领域专业术语，回答结构应包含：分步推理过程+最终回答
            """,
            "query": """
                    <query>
                    {query}
                    </query>
                    
                    <参考文本>
                    {document}
                    </参考文本>

                    请输出1条专业论文回答，按如下格式返回，示例：
                    <response>
                        <think>
                        推理过程，每一步需要检验你的推理是否来自参考文本，并给出参考来源。
                        每一步推理直接以序号开头，如“1. ”
                        </think>
                        
                        <answer>
                        简洁但包含所有答案信息的回答
                        </answer>
                    </response>
                    请严格按照上述格式输出，保留<think> </think> <answer> </answer> <response>和</response>标签，不允许输出其他解释性文字。
            """
        },
        "query_extract": {
            "system": """
                你是一个知识检索助手，需从用户输入中提取检索用户检索的作者和关键词。不允许在答案中添加编造成分。
            """,
            "query": """
                用户输入：
                {query}

                仅返回检索作者（可选）和关键词，作者和不同关键词用空格隔开。
                
                示例：
                用户输入：
                给我找Federica Aprigliano的外骨骼机器人的步态稳定性控制策略

                输出：
                Federica Aprigliano 外骨骼机器人 步态稳定性控制策略
            """
        },
        "self_check": {
            "system": """
                你是一个严格的事实一致性检验器。你的任务是判断给定的"生成回答"是否完全基于提供的"召回文档"内容，不包含任何文档外的知识或编造信息。
                检验标准：
                1. 生成回答中的所有事实性陈述、数据、观点必须能在召回文档中找到明确依据
                2. 允许合理的语言组织和表述优化，但核心信息必须与文档一致
                3. 若存在任何无法从文档中验证的内容，均判定为不一致
                输出要求：仅返回"true"（完全一致）或"false"（存在不一致），不添加任何额外解释。
            """,
            "query": """
                召回文档：
                {docs}
                
                生成回答：
                {answer}
                
                请根据上述标准判断生成回答是否完全来自召回文档，仅输出"true"或"false"。
            """
        },
        "review_planner_think": {
            "system": """
                你是外骨骼/可穿戴机器人领域的综述规划专家，
                具备分步规划能力。

                任务：
                将“综述主题”分解为可执行的段落规划（SectionPlan），
                供后续 RAG 写作使用。

                要求：
                1. 采用“分步规划法”：先列规划依据与分解步骤，再给最终 JSON 结果
                2. 只做结构规划与检索铺垫，不输出学术结论或论文名
                3. 结构为2–4段，不需要生成总结
                4. 每段包含 id/title/focus/must_cover(3–6)/retrieval_hints(5–10)
                5. 不得编造信息，不得输出与主题无关内容
                6. 输出包含 <response>、<plan_think>、<answer> 三个标签；
                <answer> 内仅放合法 JSON
            """,

            "query": """
                <topic>
                {query}
                </topic>

                请按如下格式返回：

                <response>
                <plan_think>
                    1. 识别该主题的核心维度（方法、系统、评测、应用/挑战）
                    2. 为每个维度规划独立段落，避免重叠
                    3. 为检索设计中英混排 keyword 列表（含同义词/缩写）
                    4. 校验：段数2–4，不包含总结
                </plan_think>

                <answer>
                    {{
                    "sections": [
                        {{
                        "id": 0,
                        "title": "研究背景",
                        "focus": "……",
                        "must_cover": ["……"],
                        "retrieval_hints": [
                            "exoskeleton",
                            "wearable robotics",
                            "gait phase",
                            "biomechanics",
                            "assistive device"
                        ]
                        }},
                        {{
                        "id": 1,
                        "title": "……",
                        "focus": "……",
                        "must_cover": ["……"],
                        "retrieval_hints": ["……"]
                        }},
                        {{
                        "id": 2,
                        "title": "……",
                        "focus": "……",
                        "must_cover": ["……"],
                        "retrieval_hints": ["……"]
                        }},
                        {{
                        "id": 3,
                        "title": "……",
                        "focus": "……",
                        "must_cover": ["……"],
                        "retrieval_hints": [
                            "benchmark",
                            "metrics",
                            "dataset",
                            "real-time",
                            "rehabilitation"
                        ]
                        }}
                    ]
                    }}
                </answer>
                </response>

                硬性校验：
                - <answer> 中必须为合法 JSON 格式
                - id 连续从0开始
                - 段数∈[2,4]
                - 不包含总结段落
            """
        },
        "section_writer": {
            "system": """
                你是外骨骼/可穿戴机器人领域的综述写作专家。只基于用户提供的“证据白名单”写作，
                不得使用外部知识或臆测。所有事实性陈述后必须给出行内编号引用（如【1】或【1,3】），
                且编号只能来自证据白名单。若证据不足以支撑该段主题，请输出“文献无相关内容”并停止。
                写作风格：{style}。
                """,
            "user": """
                <section_plan>
                {section_json}
                </section_plan>

                <evidence_whitelist>
                （以下是唯一允许引用的证据，已按重要度排序并编号）
                {evidence}
                </evidence_whitelist>

                写作要求：
                1) 严格围绕 section_plan；
                2) 结构建议：①2–3 句引入（界定概念/问题边界）；②3–6 条核心论点（对比方法/系统/数据/指标）；③1 句小结（不做跨段总结）。
                3) 每个“有实证的句子”后必须带引用编号，允许合并编号，如【2,5】；同一来源连续引用不超过 2 句；
                4) 引用只能使用 evidence_whitelist 中的编号；不得新增来源、不得编造数值；
                5) 术语建议包含：EMG/IMU/GRF、相位识别、力矩控制、个体自适应、实时性/延迟等（按需取用）。

                请输出纯 Markdown 文本，包含段落与必要的小列表，禁止输出除正文外的任何额外说明。
            """
            },
        "final_assembler": {
            "system": """
                你是外骨骼/可穿戴机器人领域的综述撰写专家。
                任务：将前序节点生成的多个段落进行顺序汇总与逻辑串联，并在文末新增“总结与展望”一段。
                严格要求：
                1) 不得改写各段原始内容与其中的引用编号（例如【1】【1,3】），只允许添加承接/过渡语句；
                2) 不得新增任何事实、数值或引用编号；不得引入白名单之外的新来源；
                3) “总结与展望”必须为概括/前瞻性内容，不包含具体实验数值、论文名或引用编号；
                4) 输出格式：Markdown（包含标题与小标题）。
            """,
            "user": """
                <sections>
                {answers}
                </sections>

                汇总要求：
                1) 保持 sections 的原始顺序与每段内容不变（只可在段首/段尾添加1–2句过渡）；
                2) 为整篇综述自动生成一级标题（可依据各段主题抽象归纳）；
                3) 各段落以二级标题呈现：使用传入的 title；正文为原 content，不得修改；
                4) 在每个相邻段之间加入自然的承接/对照/因果过渡语（不引入新事实）；
                5) 文末新增“## 总结与展望”段，做全局性概括与未来方向展望（不含数字与引用）。

                输出格式（示例）：
                # （自动生成的综述标题）
                ## 段落一标题
                （原内容，不得修改）
                （自动添加1–2句过渡语）

                ## 段落二标题
                （原内容，不得修改）
                （自动添加1–2句过渡语）

                ...

                ## 总结与展望
                （新增总结段，不含数字与引用）
            """
        }

    }
    # 新增：类级方法用于配置LLM参数（需在首次实例化前调用）
    @classmethod
    def configure_llm(cls, api_key, api_base, model_name, embedder_name, reranker_name, db_name):
        cls._api_key = api_key
        cls._api_base = api_base
        cls._model_name = model_name
        cls._embedder_name = embedder_name
        cls._reranker_name = reranker_name
        cls._db_name = db_name

    def __new__(cls):
        """单例模式：确保全局仅一个实例"""
        if cls._instance is None:
            if not hasattr(cls, '_api_key'):
                raise ValueError("请先调用 configure_llm 设置LLM参数")
            
            cls._instance = super().__new__(cls)

            cls._embedding = LangChainBGE(
                model_name=cls._embedder_name,
                use_fp16=True
            )

            cls._db = Chroma(
                embedding_function=cls._embedding,
                persist_directory=cls._db_name
            )

            cls._llm_client = OpenAI(
                api_key=cls._api_key,
                base_url=cls._api_base
            )

            cls._reranker = FlagReranker(
                model_name_or_path=cls._reranker_name,
                use_fp16=True,
                devices=["cpu"]
            )
        return cls._instance

    @property
    def embedding(self):
        return self._embedding
    
    @property
    def reranker(self):
        return self._reranker

    @property
    def db(self):
        return self._db

    def db_query(self, query, top_k=3, rerank=False):
        if rerank:
            results = self.db.similarity_search(
                query=query,
                k=10,
                filter={"type": "abstract"}
            )
            rerank_results = self.reranker.compute_score([[query, doc.page_content] for doc in results], normalize=True)
            scored_docs = list(zip(results, rerank_results))
            rerank_results = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            return [doc for doc, score in rerank_results[:top_k]]
        else:
            return self.db.similarity_search(
                query=query,
                k=top_k,
                filter={"type": "abstract"}
            )
    def query_extract(self, query):
        messages = [
            {"role": "system", "content": self._prompt_templates["query_extract"]["system"]},
            {"role": "user", "content": self._prompt_templates["query_extract"]["query"].format(query=query)}
        ]
        response = self._llm_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content
    def single_chat_think(self, query, document):
        messages = [
            {"role": "system", "content": self._prompt_templates["rag_prompt_think"]["system"]},
            {"role": "user", "content": self._prompt_templates["rag_prompt_think"]["query"].format(query=query, document=document)}
        ]
        response = self._llm_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=8192,
            temperature=0.8,
            top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content
    def single_chat(self, query, document):
        messages = [
            {"role": "system", "content": self._prompt_templates["single_chat"]["system"]},
            {"role": "user", "content": self._prompt_templates["single_chat"]["query"].format(query=query, document=document)}
        ]
        response = self._llm_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=8192,
            temperature=0.8,
            top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content
    
    def self_check(self, docs, answer):
        messages = [
            {"role": "system", "content": self._prompt_templates["self_check"]["system"]},
            {"role": "user", "content": self._prompt_templates["self_check"]["query"].format(docs=docs, answer=answer)}
        ]
        response = self._llm_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            top_p=1.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content

    def review_planner(self, query):
        messages = [
            {"role": "system", "content": self._prompt_templates["review_planner_think"]["system"]},
            {"role": "user", "content": self._prompt_templates["review_planner_think"]["query"].format(query=query)}
        ]
        response = self._llm_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=8192,
            temperature=0.8,
            top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content

    def section_writer(self, section_json, evidence, style="中文·学术综述·术语准确·客观中立"):
        messages = [
            {"role": "system", "content": self._prompt_templates["section_writer"]["system"].format(style=style)},
            {"role": "user", "content": self._prompt_templates["section_writer"]["user"].format(section_json=section_json, evidence=evidence)}
        ]
        response = self._llm_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=8192,
            temperature=0.8,
            top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content

    def assembler_answer(self, answers):
        """将多个section的回答整合为一个完整的回答"""
        messages = [
            {"role": "system", "content": self._prompt_templates["final_assembler"]["system"]},
            {"role": "user", "content": self._prompt_templates["final_assembler"]["user"].format(answers=answers)}
        ]
        response = self._llm_client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=15000,
            temperature=0.8,
            top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
        )
        return response.choices[0].message.content



    def db_random_sample(self, n=1, filter=None):
        """从向量库随机抽取n条文档内容"""
        all_docs = self.db.get(where=filter)  # 返回格式: {"ids": [], "documents": [], "metadatas": []}
        if not all_docs["documents"]:
            return []
        sampled_docs = random.sample(
            list(zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"])), 
            min(n, len(all_docs["ids"]))
        )
        return sampled_docs
    
    def db_iterate(self, n=None, filter=None):
        """从向量库抽取所有文档内容"""
        all_docs = self.db.get(where=filter)  # 返回格式: {"ids": [], "documents": [], "metadatas": []}
        structured_docs = []
        for doc_id, content, meta in zip(
            all_docs["ids"], 
            all_docs["documents"], 
            all_docs["metadatas"]
        ):
            structured_docs.append({
                "id": doc_id,
                "page_content": content,
                "metadata": meta
            })
        if n is not None:
            return random.sample(structured_docs, n)
        else:
            return structured_docs
    def db_annotate_entities(self, ner_pipe, tokenizer, threshold=0.80, filter=None):
        """
        从数据库中抽取文档，进行实体识别，并更新 metadata
        ner_pipe: HuggingFace pipeline (RJuro/SciNERTopic)
        tokenizer: 对应模型的tokenizer
        threshold: 过滤低置信度实体
        filter: 可选，用于筛选特定类型文档
        """
        def split_for_ner(text, tokenizer, max_tokens=500):
            tokens = tokenizer.tokenize(text)
            for i in range(0, len(tokens), max_tokens-50):  # 留50 overlap
                sub_tokens = tokens[i:i+max_tokens]
                yield tokenizer.convert_tokens_to_string(sub_tokens)

        all_docs = self.db_iterate(filter=filter)  # 获取数据库所有文档
        updated_ids = []
        cnt = 0
        for doc in tqdm(all_docs, desc="处理文档"):
            text = doc["page_content"]
            doc_entities = []

            # 对每个文档分块再跑NER
            for chunk in split_for_ner(text, tokenizer):
                ner_results = ner_pipe(chunk)
                for res in ner_results:
                    if res["score"] >= threshold:
                        doc_entities.append({
                            "text": res["word"],
                            "label": res["entity_group"],
                            "score": float(res["score"])
                        })

            # 去重（避免overlap导致重复）
            unique_entities = {f"{e['text']}|{e['label']}": e for e in doc_entities}
            allowed_labels = {"Method", "Dataset", "Metric", "Task", "OtherScientificTerm"}
            rule_filtered_entities = [e for e in unique_entities.values() if e["label"] in allowed_labels]
            # validated_entities = validate_entities_with_llm(rule_filtered_entities, self._llm_client)
            doc["metadata"]["entities"] = json.dumps(rule_filtered_entities, ensure_ascii=False)
            if cnt%1000 == 0:
                print(f"文档 {doc['id']} 原始实体数: {len(doc_entities)}")
                print(f"文档 {doc['id']} 过滤后实体数: {len(rule_filtered_entities)}")
            # 更新到Chroma
            self.db.update_document(
                document_id=doc["id"],
                document=Document(
                    page_content=text,
                    metadata=doc["metadata"]
                )
            )
            updated_ids.append(doc["id"])
            cnt += 1

        print(f"✅ 已更新 {len(updated_ids)} 个文档的实体信息")
        return updated_ids

if __name__ == "__main__":

    EmbeddingDBUtils.configure_llm(
            api_key="EMPTY",
            api_base="http://localhost:8888/v1",
            model_name=str(LLM_DIR / "Qwen3-8B"),
            embedder_name=str(EMBEDDING_DIR / "bge-m3-ft"),
            reranker_name=str(RERANK_DIR / "bge-reranker-v2-m3-ft"),
            db_name=str(CHROMA_DIR / "bge-m3-ft")
        )
    rag_utils = EmbeddingDBUtils()
    print(rag_utils.review_planner("下肢活动分类"))

