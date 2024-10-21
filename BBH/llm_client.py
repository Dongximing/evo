import json
import logging
import os
import atexit
import requests
import sys
from tqdm import tqdm
import openai
import backoff
from termcolor import colored
import time
from utils import read_yaml_file, batchify
from openai import OpenAI
import os
KEY = ""
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",
                                       KEY))

def extract_seconds(text, retried=5):
    words = text.split()
    for i, word in enumerate(words):
        if "second" in word:
            return int(words[i - 1])
    return 60


def form_request(data, type, **kwargs):
    if "davinci" in type:
        request_data = {
            "prompt": data,
            "max_tokens": 1000,
            "top_p": 1,
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
            "logprobs": None,
            "stop": None,
            **kwargs,
        }
    else:
        print("37----->",data)
        assert isinstance(data, str)
        # print(data)
        messages_list = [{"role": "system", "content": "Follow the given examples and answer the question."}]
        messages_list.append({"role": "user", "content": data})
        request_data = {
            "messages": messages_list,
            # "temperature": 0,
            # "engine":'gpt-35-turbo',
            **kwargs
        }
    # print(request_data)
    return request_data


def llm_init(auth_file="../auth.yaml", llm_type='davinci', setting="default"):
    auth = read_yaml_file(auth_file)[llm_type][setting]
    try:
        openai.api_type = auth['api_type']
        openai.api_base = auth["api_base"]
        openai.api_version = auth["api_version"]
    except:
        pass
    openai.api_key = auth["api_key"]
    return auth

def turbo_query(request_data, **kwargs):
    print("request_data",request_data)

    while True:
        retried = 0
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                # engine="gpt-35-turbo",
                messages=[
                        {"role": "system", "content": "Follow the given examples and answer the question."},
                        {"role": "user", "content": request_data},
                    ],
                logprobs=True,
                )
            # print("\n\n\n")
            # print("-"*50)
            # print("request_data",response)
            # print("-"*50)
            break
        except Exception as e:
            error = str(e)
            print("retring...", error)
            second = extract_seconds(error, retried)
            retried = retried + 1
            time.sleep(second)

    return response.choices[0].message.content, response.choices[0].logprobs

def davinci_query(data,client,**kwargs):
    retried=0
    request_data = {
            "prompt": data,
            "max_tokens": 1000,
            "temperature": 0,
            **kwargs,
        }
    while True:
        try:
            response = openai.Completion.create(**request_data)
            response = response["choices"]
            response = [r["text"] for r in response]
            break
        except Exception as e:
            error = str(e)
            print("retring...", error)
            second = extract_seconds(error, retried)
            retried = retried + 1
            time.sleep(second)
    return response

def llm_query(data, client, type, task, **config):
    hypos = []
    response = None
    model_name = "turbo"
    # batch
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",
                                       KEY))
    print("data in 112",data)
    if isinstance(data, list):
        batch_data = batchify(data, 20)
        print("data in 114", data)
        for batch in tqdm(batch_data):

            if "davinci" in type:
                print("batch in 117", batch)
                retried = 0
                request_data = form_request(batch, model_name, **config)
                # print(request_data)
                while True:
                    try:
                        response = openai.Completion.create(**request_data)
                        response = response["choices"]
                        response = [r["text"] for r in response]
                        break
                    except Exception as e:
                        error = str(e)
                        print("retring...", error)
                        second = extract_seconds(error, retried)
                        retried = retried + 1
                        time.sleep(second)
                    
            else:
                response = []
                for data in batch:
                    request_data = form_request(data, type, **config)
                    print("BBH/llm_client.py:151",request_data)
                    del request_data['api_key']
                    while True:
                        try:
                            print(client)
                            result = client.chat.completions.create(**request_data)
                            result = result.choices[0].message.content
                            response.append(result)
                            print("BBH/llm_client.py:156",result)
                            break
                        except Exception as e:
                            error = str(e)
                            print("retring...", error)
                            second = extract_seconds(error, retried)
                            retried = retried + 1
                            time.sleep(second)

            print(response)
            if task:
                results = [str(r).strip().split("\n\n")[0] for r in response]
            else:
                results = [str(r).strip() for r in response]
            # print(results)
            # results = [str(r['text']).strip() for r in response]
            # print(results)
            hypos.extend(results)
    else:
        retried = 0
        while True:
            try:
                print(type)
                result = ""
                if "turbo" in type or 'gpt4' in type:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Follow the given examples and answer the question."},
                            {"role": "user", "content": data},
                        ],
                        logprobs=True,
                    )
                    result = response.choices[0].message.content

                    break
                else:
                    request_data = form_request(data, type=type, **config)
                    response = client.chat.completions.create(**request_data)
                    result = response.choices[0].message.content
                    # result = result['text']
                    result = result.strip()
                break
            except Exception as e:
                error = str(e)
                print("retring11111...", error)
                second = extract_seconds(error, retried)
                retried = retried + 1
                time.sleep(second)
        if task:
            result = result.split("\n\n")[0]

        hypos = result
    return hypos


def paraphrase(sentence, client, type, **kwargs):
    if isinstance(sentence, list):
        resample_template = [
            f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{s}\nOutput:"
            for s in sentence
        ]
    else:
        resample_template = f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{sentence}\nOutput:"
    print(resample_template)
    results = llm_query(resample_template, client, type, False, **kwargs)
    return results

if __name__ == "__main__":
    llm_client = None
    llm_type = 'davinci'
    start = time.time()
    data =  [""" Q: Tom bought a skateboard for $ 9.46 , and spent $ 9.56 on marbles . Tom 
also spent $ 14.50 on shorts . In total , how much did Tom spend on toys ?                                                 
A: Let's think step by step. """]
    config = llm_init(auth_file="auth.yaml", llm_type=llm_type, setting="gcr1")
    # para = turbo_query('hi')
    para = paraphrase(data, llm_client, llm_type, **config)
    print(para)
    end = time.time()
    print(end - start)
