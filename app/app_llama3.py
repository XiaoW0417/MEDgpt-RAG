import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

# 页面设置
st.set_page_config(page_title="🧠 RAG 医疗问答系统", layout="wide")
st.title("🧬 本地 RAG + Llama3.1-8B（Ollama） 医疗问答系统")

# 加载向量数据库
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.load_local("D:\\Python_projects\\LLM\\MEDgpt-RAG\\data\\vect_medical_knowledge", embedding_model, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt 模板
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

# LLM 设置（启用流式）
llm = ChatOllama(model="llama3.1:8b", temperature=0.7, stream=True)

# 用户输入
query = st.text_input("💬 请输入你的医学问题：", placeholder="例如：什么是高血压的常见并发症？")

if query:
    with st.spinner("🧠 正在分析并生成回答..."):

        # 1. 检索上下文
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. 构造 prompt
        prompt = prompt_template.format(context=context, question=query)

        # 3. 流式输出
        st.markdown("### 📌 回答：")
        output_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(prompt):
            full_response += chunk.content
            output_placeholder.markdown(full_response)

        # 4. 展示原始文档
        with st.expander("📚 检索到的上下文文档"):
            for i, doc in enumerate(docs):
                st.markdown(f"#### 文档 {i+1}")
                st.code(doc.page_content[:1000], language='markdown')
