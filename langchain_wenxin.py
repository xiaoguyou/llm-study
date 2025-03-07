from wenxin_llm import Wenxin_LLM
from dotenv import find_dotenv, load_dotenv
import os

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
wenxin_api_key = os.environ["QIANFAN_ACCESS_KEY"]
wenxin_secret_key = os.environ["QIANFAN_SECRET_KEY"]


llm = Wenxin_LLM(api_key=wenxin_api_key, secret_key=wenxin_secret_key, system="你是一个助手！")


# llm.invoke("你好，请你自我介绍一下！")

print(llm(prompt="你好，请你自我介绍一下！"))
