from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os, json

# 1. 加载所有 .md 文件
loader = DirectoryLoader(
    path=".",  # 当前目录
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)
md_documents = loader.load()

json_documents = []
with open("med_qa.txt", "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            text = f"{item['context'].strip()}\n{item['target'].strip()}"
            doc = Document(page_content=text)
            json_documents.append(doc)
        except Exception as e:
            print(f"跳过出错行: {e}")
            continue

all_documents = md_documents #+ json_documents

# 2. 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(all_documents)

# 3. 加载 embedding 模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 4. 构建 FAISS 向量数据库
vectorstore = FAISS.from_documents(docs, embedding_model)

# 5. 保存数据库
save_path = "./vect_medical_knowledge"
os.makedirs(save_path, exist_ok=True)
vectorstore.save_local(save_path)

print(f"成功保存向量数据库到：{save_path}")
