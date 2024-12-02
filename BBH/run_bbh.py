# evaluating GPT-3.5 turbo model on BBH
import logging
from operator import index

import openai
import json
import numpy as np
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
from utils import extract_ans, batchify
from llm_client import turbo_query, davinci_query
import tiktoken
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MULTIPLE_CHOICE_TASKS = [
        'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table', 
        'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects', 
        'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation', 
        'salient_translation_error_detection', 'reasoning_about_colored_objects', 
]
FREE_FORM_TASKS = [
        'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding', 
        'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies', 
]

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def create_dataset(mode, task_prompt, cot_prompt, eval_data,demon=1):
    questions = []
    prompt_qs = []
    answers= []
    # print("BBH/run_bbh.py:38",eval_data)
    # print("BBH/run_bbh.py:39", cot_prompt)

    for q_ in eval_data:
        task_prompt = task_prompt.replace('<prompt>', cot_prompt)
        if demon: 
            q = '\n\nQ: ' + q_['input']
            prompt_q = task_prompt + q + f"\nA: {cot_prompt}"
        else:
            q = 'Q: ' + q_['input']
            prompt_q = q + f"\nA: {cot_prompt}"
        questions.append(q)
        prompt_qs.append(prompt_q)
        if mode == 'multiple_choice':
            a = q_['target'][1]
        elif mode == 'free_form':
            a = q_['target']
        answers.append(a)
    return prompt_qs, questions,answers
def create_parallel_dataset(mode, task_prompt, cot_prompts, eval_data,demon=1):
    questions = []
    prompt_qs = []
    answers= []
    # print("BBH/run_bbh.py:38",eval_data)
    print("BBH/run_bbh.py:39", cot_prompts)
    # print("BBH/run_bbh.py:40",task_prompt)
    # print("BBH/run_bbh.py:64", eval_data)
    for cot_prompt in cot_prompts:
        for q_ in eval_data:
            task_prompt = task_prompt.replace('<prompt>', cot_prompt)
            if demon:
                q = '\n\nQ: ' + q_['input']
                prompt_q = task_prompt + q + f"\nA: {cot_prompt}"
            else:
                q = 'Q: ' + q_['input']
                prompt_q = q + f"\nA: {cot_prompt}"
            questions.append(q)
            prompt_qs.append(prompt_q)
            if mode == 'multiple_choice':
                a = q_['target'][1]
            elif mode == 'free_form':
                a = q_['target']
            answers.append(a)
    # print(f"answers ---------------------------------->{answers}")
    return prompt_qs, questions,answers
def create_request(custom_id, user_message):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo-0125",
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "logprobs": True,
            "top_logprobs": 1,

        },
    }
def inference_openai(sentences,seed):
    import json

    requests = [create_request(f"request-{i + 1}", msg) for i, msg in enumerate(sentences)]

    file_path = f'api_requests_{seed}.jsonl'

    # Write each request to the .jsonl file
    with open(file_path, 'w') as file:
        for request in requests:
            # Convert each dictionary to a JSON string
            json_line = json.dumps(request)
            # Write the JSON string to the file followed by a newline
            file.write(json_line + '\n')

    print(f"File saved successfully to {file_path}")
    batch_input_file = client.files.create(
        file=open(f"{file_path}", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    print(batch_input_file_id)

    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    batch_id = client.batches.list().first_id
    import time
    running = True
    while running == True:
        status = client.batches.retrieve(batch_id)
        print(f"Current status: {status}")
        if status.status == 'completed':
            print("Batch processing is complete.")
            running = False
        elif status.status == 'failed':
            print("Batch processing failed.")
            running = False
        else:
            print("Batch still processing. Waiting...")
            time.sleep(10)  # wait for 10 seconds before checking again

    file_response = client.files.content(status.output_file_id)
    response_file = f"response_file_{seed}.jsonl"
    with open(response_file, 'w') as file:
            file.write(file_response.text)
    import json
    output_cost = 0
    list_top20_logprobs = []
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    all_data = []
    with open(f'response_file_{seed}.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            all_data.append(data)
    # print("BBH/run_bbh.py:159",all_data)
    # all_data.sort(key=lambda x: x['custom_id'])
    # print("BBH/run_bbh.py:161", all_data)
    responses = []
    for data in all_data:
        top_twenty_logprobs = data["response"]["body"]["choices"][0]["logprobs"]["content"]
        response = data["response"]["body"]["choices"][0]["message"]["content"]
        responses.append(response)

        output_cost += len(encoding.encode(response))
        list_top20_logprobs.append(top_twenty_logprobs)
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)
        print("File deleted successfully.")
    else:
        # File does not exist
        print("File not found.")
    if os.path.exists(response_file):
        # Delete the file
        os.remove(response_file)
        print("File deleted successfully.")
    else:
        # File does not exist
        print("File not found.")
    # print('----------------------------------------------------------------')
    # print(list_top20_logprobs)
    # breakpoint()

    return list_top20_logprobs, output_cost,responses


def first_step_parallel_pool(task, task_prompt,cot_prompt,eval_data, client, model_index,logger,demon ,seed,**kwargs):
    mode = 'multiple_choice' if task in MULTIPLE_CHOICE_TASKS else 'free_form'
    prompt_qs, questions, answers = create_parallel_dataset(mode, task_prompt, cot_prompt, eval_data, demon)
    print(prompt_qs)
    list_top20_logprobs, output_cost,responses = inference_openai(prompt_qs,seed)
    return list_top20_logprobs, output_cost,responses,answers



def eval_task(task, task_prompt,cot_prompt,eval_data, client, model_index,logger,demon, anchor, discrete, seed, **kwargs):
    # for task in tasks:
    # print('Testing %s ...' % task)
    correct = 0
    mode = 'multiple_choice' if task in MULTIPLE_CHOICE_TASKS else 'free_form'
    print_first = True



    if anchor:

        # prompt_qs, questions, answers = create_parallel_dataset(mode, task_prompt, cot_prompt, eval_data, demon)
        # print("BBH/run_bbh.py:195",len(prompt_qs),len(questions),len(answers))
        score = np.empty((0, 4))
        list_top20_logprobs, output_cost, responses,answers =first_step_parallel_pool(task, task_prompt,cot_prompt,eval_data, client, model_index,logger,demon,seed,**kwargs)
        logger.info(f"BBH/run_bbh.py:215   {len(list_top20_logprobs)} .....{len(responses)}.......{len(answers)}")
        for index, list_top20_logprob in enumerate(list_top20_logprobs):

            ans_ = extract_ans(responses[index], mode)
            logger.info(
                f"BBH/run_bbh.py:217--------model res -----{responses[index]} .........answer .......{answers[index]}.....{index}.......ans.....{ans_}......all the data.{eval_data[index]}")
            logit_matrix = np.zeros(4)
            if ans_ == answers[index]:
                if not discrete:
                    logger.info(f"*****************kkkkkddddd********{answers[index]}*******************************************\n\n")
                    if answers[index] == "ent":
                        assert list_top20_logprob[-3]["token"] == " entail"
                        logit_matrix[0] = list_top20_logprob[-3]["logprob"]
                        logger.info(f"*************************kd*******************************************\n\n")
                    elif answers[index] == "neutral":
                        assert list_top20_logprob[-2]["token"] == " neutral"
                        logit_matrix[1] = list_top20_logprob[-2]["logprob"]
                        logger.info(f"*************************kd*******************************************\n\n")
                    elif answers[index] == "contr":
                        assert list_top20_logprob[-2]["token"] == " contradiction"
                        logit_matrix[2] = list_top20_logprob[-2]["logprob"]
                        logger.info(f"***************************kd*****************************************\n\n")


                correct += 1

            score = np.vstack((score, logit_matrix))
        accuracy = correct / len(eval_data)
        logger.info(f"BBH/run_bbh.py:214   accuracy {accuracy}--------------------->{correct}")
        logger.info(f"********************************************************************\n\n")
        logger.info(f"BBH/run_bbh.py:238   score {score}")

        return accuracy, score
    prompt_qs, questions, answers = create_dataset(mode, task_prompt, cot_prompt, eval_data, demon)
    # print("BBH/run_bbh.py:212",prompt_qs)
    if 'turbo' in model_index:
        for i in tqdm(range(len(prompt_qs))):
            prompt_q = prompt_qs[i]
            q = questions[i]
            a = answers[i]

        # for prompt_q,q,a in tqdm(zip(prompt_qs, questions,answers)):
            ans_model,logits = turbo_query(prompt_q, temperature=0,**kwargs)
            ans_ = extract_ans(ans_model, mode)
            logger.info(f"ans_ ------------------->{ans_}")
            logger.info(f"real_ans_ ------------------->{a}")
            if print_first:
                logger.info('First prompt: ')
                logger.info(prompt_q)
                logger.info("first answer: ")
                logger.info(ans_model)
                logger.info(ans_)
                print_first = False
            logit_matrix = np.zeros(3)
            if ans_ == a:
                for item in logits.content:
                    print("item",item)
                    if item.token.strip() == a:
                        if a == "yes":
                            logit_matrix[0] = item.logprob
                        else:
                            logit_matrix[1] = item.logprob
                correct += 1

            score = np.vstack((score, logit_matrix))
        logger.info(f"score -----------size {score.shape}")


    accuracy = correct / len(eval_data)
    logger.info(f"prompt_qs in 96  accuracy----------------------> {accuracy}")
    print('%s acc %.4f' % (task, correct / len(eval_data)))
    return accuracy, score

