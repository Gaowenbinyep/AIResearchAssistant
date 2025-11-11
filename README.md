# AI Research Assistant


> 面向外骨骼控制 / 康复研究场景的本地化检索增强生成（RAG）系统  
> 结合自建文献向量库、BGE 系列模型与本地部署的 Qwen3-8B，支持科研问答、综述写作与结构化引用

## ✨ 功能亮点
- **纯离线可用**：嵌入模型、重排序模型、LLM 与 Chroma 向量库均可本地部署，保障数据私密性
- **双工作流**：普通问答与综述写作并行支持，可自动规划段落、分段生成并合成报告
- **事实自检**：回答流经 `self_check` 节点进行一致性校验，确保输出可追溯
- **工具化维护**：`Agent/rag_tools.py` 提供构建、重编码、抽样校验等工具，便于扩库与清洗
- **Agentic RAG**：封装 `local_search` / `web_search`，当本地知识不足时可自动切换在线检索

## 🗂️ 目录结构
```text
RAG/
├─ Agent/                      # LangGraph 节点与工作流
│  ├─ agent_tools.py           # 工具封装 + EmbeddingDBUtils
│  ├─ nodes.py                 # 提问、检索、生成、自检、综述写作
│  ├─ prompts.py               # 写作代理提示模板
│  ├─ utils.py                 # 综述规划解析与校验
│  ├─ workflow.py              # LangGraph 状态机定义
│  └─ rag_tools.py             # LLM / 嵌入 / Reranker 封装
├─ Chroma_db/                  # Chroma 向量库及快照
│  └─ bge-m3-ft/               # 已持久化的向量库
├─ Bert_model/                 # 本地 BGE 系列模型
│  ├─ embedding_model/         # BGE-m3 及微调版本
│  └─ reranker_model/          # BGE reranker
├─ LLM/Qwen3-8B/               # vLLM 使用的 Qwen3-8B 权重
├─ Data/                       # 数据样例与辅助脚本
├─ Papers/                     # PDF 文献目录
├─ logs/                       # 运行与部署日志
├─ answer.md                   # 最近一次综述写作结果
├─ main.py                     # CLI 入口
├─ Test.py                     # Ragas 异步评测脚本
└─ model_deploy.sh             # vLLM 启动脚本示例
```

## 🚀 快速开始

### 1. 环境准备
| 依赖 | 建议配置 |
| --- | --- |
| 操作系统 | Linux / macOS / Windows（推荐使用 WSL2 运行推理服务） |
| Python | 3.10 及以上 |
| GPU | ≥24 GB 显存运行 Qwen3-8B 更流畅，可根据显存调整推理参数 |

安装基础依赖：
```bash
pip install -U \
  langchain langchain-community langchain-chroma langgraph \
  openai vllm flagembedding chromadb \
  unstructured[local-pdf] pillow pandas tqdm \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. 模型与数据准备
1. 将 Qwen3-8B 权重放置于 `LLM/Qwen3-8B`（如自定义目录需在配置中同步修改）  
2. 解压 BGE 嵌入模型与 reranker 至 `Bert_model/embedding_model/`、`Bert_model/reranker_model/`  
3. 将 PDF 文献放入 `Papers/`，向量库构建时会自动读取

### 3. 构建向量库
```bash
python - <<'PY'
from Agent.rag_tools import db_build
db_build("Papers")  # 读取 Papers/ 下的 PDF，切分后写入 Chroma
PY
```
仅替换嵌入模型时，可使用 `db_reencode(existing_db_path, new_embedding_model_path)` 快速重编码。

### 4. 启动服务
1. 启动本地 LLM（默认监听 `http://localhost:8888/v1`）：
   ```bash
   bash model_deploy.sh
   ```
2. 运行 CLI 客户端（首次执行会引导配置 LLM / 嵌入 / 向量库路径）：
   ```bash
   python main.py
   ```

## 🧭 使用说明
- **普通问答**：直接输入问题，系统自动抽取检索关键词、召回文献、生成包含思维链的回答并完成自检
- **结束对话**：输入 `exit` / `quit` / `end`
- **继续提问**：自检通过后按提示输入 `y` 继续，其他输入结束会话
- **综述写作**：以 `写综述：` 开头描述需求（例：`写综述：下肢外骨骼步态控制最新进展`），系统将：
  1. `planner_node` 生成 2-4 个段落规划
  2. `section_agent_writer_node` 逐段自动检索与写作，如本地知识缺失会尝试 Web 搜索
  3. `assembler_answer_node` 汇总段落并输出到 `answer.md`

## 🧪 评测脚本（`Test.py`）
- 复用 `Agent/agent_tools.py` 中的 `EmbeddingDBUtils` 与 `LangChainBGE`，结合 DashScope LLM 计算 faithfulness、answer relevancy、answer accuracy 等指标
- `rag_work` 使用 `asyncio` + 信号量并发调度检索与生成，默认读取 `Data/Train/llm_test_data.jsonl`
- 评测细节写入 `./test_results/evaluation_details.csv`，汇总指标直接输出在控制台
- 通过 `think=True/False` 控制是否启用 Chain-of-Thought 提示，可根据显存或时延调整信号量大小

运行：
```bash
python Test.py
```
如需评测其他数据集，修改脚本中的 `test_data_path` 或替换指标配置即可。

## 🧱 LangGraph 工作流
| 节点 | 职责 |
| --- | --- |
| `get_user_input` | 管理用户输入、退出与继续会话状态 |
| `extract_query` | 抽取检索关键词，综述模式直接保留原始请求 |
| `document_retrieve` | 调用 Chroma + BGE reranker 获取高相关文档 |
| `generate_answer` | 使用思维链提示生成答案 |
| `self_check` | 对生成内容进行事实一致性校验 |
| `check_continue` | 判断是否继续会话 |
| `planner` | 综述模式输出段落规划 JSON |
| `section_agent_writer` | Agent 自动检索与写作，必要时触发 Web 搜索 |
| `assembler_answer` | 汇总段落并生成 Markdown 综述 |

普通问答流：`extract_query → document_retrieve → generate_answer → self_check`  
综述写作流：`extract_query → planner → section_agent_writer → assembler_answer`

## 🔧 可定制项
- **提示模板**：集中维护于 `Agent/rag_tools.py` 与 `_prompt_templates`，可按领域定制
- **模型 / 端点**：修改 `EmbeddingDBUtils.configure_llm` 参数即可更换 LLM、嵌入模型或向量库路径
- **工作流扩展**：在 `Agent/workflow.py` 以 LangGraph 方式声明节点及边，新增能力只需实现节点函数并接入
- **实体标注**：`db_annotate_entities` 支持结合 HuggingFace NER 模型为向量库添加结构化元数据

## ❓ 常见问题
- **提示未配置 LLM**：使用 `EmbeddingDBUtils` 前需先调用 `configure_llm`；`main.py` 给出了完整示例
- **`self_check` 持续返回 `false`**：说明生成内容与文档不一致，可提高检索 `top_k`、检查语料覆盖或调低生成温度
- **vLLM 端口冲突**：修改 `model_deploy.sh` 中的 `--port`，并同步更新 `nodes.py` / `workflow.py` 的 `api_base`
- **PDF 解析异常**：优先准备可搜索文本版 PDF，或调整 `RecursiveCharacterTextSplitter` 的 chunk size / overlap

---

欢迎基于本仓库进行二次开发，或集成更多检索 / 推理能力。如需调用示例，可参考 `Agent/rag_tools.py` 末尾的代码片段。祝研究顺利 🎯
