import streamlit as st
import time
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bert_score import score as bert_score

# ----------------------------
# 页面设置
# ----------------------------
st.set_page_config(page_title="🧠 RAG 医疗问答系统", layout="wide")
st.title("🧬 本地部署的多模型RAG医疗问答系统")

# ----------------------------
# 模型选择器
# ----------------------------
model_options = ["deepseek-r1:8b", "qwen3:8b", "llama3.1:8b", "gemma3:latest"]
selected_model = st.selectbox("🧠 选择本地大模型：", model_options, index=0)
## 请改为你的实际路径
vect_path = "D:\\Python_projects\\LLM\\MEDgpt-RAG\\data\\vect_medical_knowledge"

# ----------------------------
# 加载向量数据库
# ----------------------------
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.load_local(vect_path, embedding_model, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 召回 10 条文档

# ----------------------------
# 加载 Reranker 模型
# ----------------------------
@st.cache_resource
def load_reranker():
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    return tokenizer, model

tokenizer, reranker_model = load_reranker()

def rerank(query, docs, top_k=5):
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = tokenizer.batch_encode_plus(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze()
    ranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked_docs[:top_k]

# ----------------------------
# Prompt 模板
# ----------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
你是一位医学知识助手。请根据以下上下文内容回答问题，不要加入任何其他信息。如果上下文中没有答案，请回答“我无法确定答案”。

【上下文】
{context}

【问题】
{question}

【回答】
""".strip()
)

# ----------------------------
# LLM 初始化
# ----------------------------
llm = ChatOllama(model=selected_model, temperature=0.7, stream=True)

# ----------------------------
# 用户输入
# ----------------------------
query = st.text_input("💬 请输入你的医学问题：", placeholder="例如：什么是高血压的常见并发症？")
reference_answer = st.text_area("🎯 可选：输入参考答案以评估（用于 BERTScore）", height=100)

if query:
    with st.spinner("🧠 正在分析并生成回答..."):
        start_time = time.time()

        # 1. 检索 & rerank
        retrieved_docs = retriever.get_relevant_documents(query)
        reranked_docs = rerank(query, retrieved_docs, top_k=3)

        # 2. 构造 Prompt 并推理
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        prompt = prompt_template.format(context=context, question=query)

        st.markdown("### 📌 回答：")
        output_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(prompt):
            full_response += chunk.content
            output_placeholder.markdown(full_response)

        # 3. 时间统计
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        st.success(f"✅ 推理完成，耗时 {elapsed_time} 秒 | 使用模型：`{selected_model}`")

        # 4. 显示上下文文档
        with st.expander("📚 检索到的上下文文档（Reranked）"):
            for i, doc in enumerate(reranked_docs):
                st.markdown(f"#### 文档 {i+1}")
                st.code(doc.page_content[:1000], language='markdown')

        # 6. BERTScore（需要用户输入参考答案）
        if reference_answer.strip():
            st.markdown("### 🔍 BERTScore 评估")
            P, R, F1 = bert_score([full_response], [reference_answer], lang="zh", verbose=False)
            st.write(f"**Precision:** {P[0]:.4f}")
            st.write(f"**Recall:** {R[0]:.4f}")
            st.write(f"**F1:** {F1[0]:.4f}")
