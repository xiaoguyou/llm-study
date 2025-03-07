import sys
sys.path.append("../C3 搭建知识库") # 将父目录放入系统路径中

# 使用百度千帆 Embedding
from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from langchain.vectorstores.chroma import Chroma


from dotenv import load_dotenv, find_dotenv
import os

from wenxin_llm import Wenxin_LLM
from dotenv import find_dotenv, load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


_ = load_dotenv(find_dotenv())    # read local .env file
qianfan_ak = os.environ.get("QIANFAN_ACCESS_KEY")  # 使用您之前设置的环境变量名
qianfan_sk = os.environ.get("QIANFAN_SECRET_KEY")  # 使用您之前设置的环境变量名


# 定义 Embeddings

embedding = QianfanEmbeddingsEndpoint(
    qianfan_ak = qianfan_ak,
    qianfan_sk = qianfan_sk
)


# 向量数据库持久化路径
persist_directory = 'data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")

# question = "什么是prompt engineering?"
# docs = vectordb.similarity_search(question,k=3)
# print(f"检索到的内容数：{len(docs)}")

# for i, doc in enumerate(docs):
#     print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")



# question = "什么是prompt engineering?"
# docs = vectordb.similarity_search(question,k=3)
# # print(f"检索到的内容数：{len(docs)}")

# # for i, doc in enumerate(docs):
# #     print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")


llm = Wenxin_LLM(api_key=qianfan_ak, secret_key=qianfan_sk)


# llm.invoke("你好，请你自我介绍一下！")

# print(llm(prompt="你好，请你自我介绍一下！"))

from langchain.prompts import PromptTemplate

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
最多使用三句话。尽量使答案简明扼要。总是在回答的最后说"谢谢你的提问！"
{context}
问题: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

question_1 = "什么是南瓜书？"
question_2 = "王阳明是谁？"

# result = qa_chain({"query": question_1})
# print("大模型+知识库后回答 question_1 的结果：")
# print(result["result"])
# print("大模型自己的回答：" + llm(prompt=question_1))

# result = qa_chain({"query": question_2})
# print("大模型+知识库后回答 question_2 的结果：")
# print(result["result"])
# print("大模型自己的回答：" + llm(prompt=question_2))



memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)

retriever=vectordb.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)
question = "为什么这门课需要教这方面的知识？"
result = qa({"question": question})
print(result['answer'])

