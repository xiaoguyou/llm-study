import nltk
import os
from dotenv import load_dotenv, find_dotenv

# 使用百度千帆 Embedding
from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint

from langchain.vectorstores.chroma import Chroma


# 加载环境变量
_ = load_dotenv(find_dotenv())
# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
qianfan_ak = os.environ.get("QIANFAN_ACCESS_KEY")  # 使用您之前设置的环境变量名
qianfan_sk = os.environ.get("QIANFAN_SECRET_KEY")  # 使用您之前设置的环境变量名


# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = 'data_base/knowledge_db'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
# print(file_paths[:3])


from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []

for file_path in file_paths:

    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))
# 下载文件并存储到text
texts = []

for loader in loaders: 
    texts.extend(loader.load())
    

text = texts[1]
# print(f"每一个元素的类型：{type(text)}.", 
#     f"该文档的描述性数据：{text.metadata}", 
#     f"查看该文档的内容:\n{text.page_content[0:]}", 
#     sep="\n------\n")


from langchain.text_splitter import RecursiveCharacterTextSplitter

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(texts)


embedding = QianfanEmbeddingsEndpoint(
    qianfan_ak = qianfan_ak,
    qianfan_sk = qianfan_sk
)

# 定义持久化路径
persist_directory = 'data_base/vector_db/chroma'

# 检查向量数据库是否已存在
if os.path.exists(persist_directory):
    # 如果存在，直接加载现有数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
else:
    # 如果不存在，创建新的向量数据库
    vectordb = Chroma.from_documents(
        documents=split_docs[:20], # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
        embedding=embedding,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")


question="什么是大语言模型"     
# 相似度检索
sim_docs = vectordb.similarity_search(question,k=3) # k=3 表示返回最相似的 3 个内容
print(f"检索到的内容数：{len(sim_docs)}")

for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

# MMR检索
mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)

for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
