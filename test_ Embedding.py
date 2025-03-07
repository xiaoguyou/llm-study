import requests
import json
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


def wenxin_embedding(text: str):
    # 获取环境变量 wenxin_api_key、wenxin_secret_key
    api_key = os.environ.get('QIANFAN_ACCESS_KEY')
    secret_key = os.environ.get('QIANFAN_SECRET_KEY')

    # 使用API Key、Secret Key向https://aip.baidubce.com/oauth/2.0/token 获取Access token
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret={1}".format(
        api_key, secret_key)
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)



    # 通过获取的Access token 来embedding text
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + str(response.json().get("access_token"))
    input = []
    input.append(text)
    playload = json.dumps({
        "input": input,
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=playload)

    return json.loads(response.text)



text = '测试向量'
result = wenxin_embedding(text)

print('本次embedding id为：{}'.format(result['id']))
print('本次embedding产生时间戳为：{}'.format(result['created']))
print('返回的embedding类型为:{}'.format(result['object']))
print('embedding长度为：{}'.format(len(result['data'][0]['embedding'])))
print('embedding（前10）为：{}'.format(result['data'][0]['embedding'][:10]))

