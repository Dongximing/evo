import random
import random
import numpy as np
import yaml
import logging
from logging.handlers import TimedRotatingFileHandler
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast


def find_lowest_similarity_pair(BP):
    similarity_matrix = cosine_similarity(BP)

    np.fill_diagonal(similarity_matrix, np.inf)

    min_sim = np.min(similarity_matrix)
    indices = np.where(similarity_matrix == min_sim)
    first_index = indices[0][0]
    second_index = indices[1][0]

    return first_index, second_index

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
def batchify(data, batch_size=20):
    batched_data = []
    for i in range(0, len(data), batch_size):
        batched_data.append(data[i:i + batch_size])
    return batched_data

def setup_log(log_path, log_name="basic"):
    print("Setting up log for", log_name)
    logger = logging.getLogger(log_name)
    if not logger.handlers:
        # log_path = os.path.join("logs", log_name)
        logger.setLevel(logging.DEBUG)
        file_handler = TimedRotatingFileHandler(
            filename=log_path, when="MIDNIGHT", interval=1, backupCount=30
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        stream_handler = logging.StreamHandler()
        # formatter = logging.Formatter("[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s")
        formatter = logging.Formatter("[%(asctime)s] - %(message)s")

        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    return logger

def read_lines(file_, sample_indices=None):
    ret = []
    if sample_indices:
        sample_indices.sort()
        with open(file_, 'r') as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    ret.append(line.rstrip())
        return ret
    else:
        with open("/mnt/hdd-data/shaowei/data_selection/evo/BBH/prompts.txt", 'r') as f:
            lines = f.readlines()
        return [line.rstrip() for line in lines]

def k_init_pop(initial_mode, init_population, k):
    if initial_mode == "topk":
        population = [i for i in init_population[:k]]
    elif initial_mode == "para_topk":
        population = [i for i in init_population[: k // 2]]
    elif initial_mode == "para_bottomk":
        population = [i for i in init_population[-k // 2 :]]
    elif initial_mode == "para_randomk":
        population = random.sample(init_population, k // 2)
    elif initial_mode == "randomk":
        population = random.sample(init_population, k)
    elif initial_mode == "bottomk":
        population = [i for i in init_population[-k:]]
    return population

def get_final_prompt(text):
    parts = text.split("<prompt>")
    if len(parts) > 1:
        prompt = parts[-1].split("</prompt>")[0]
        prompt = prompt.strip()
        return prompt
    else:
        if text.startswith("\"") and text.endswith("\""):
            text = text[1:-1]
        return text

def extract_ans(ans, mode):
    print("ans----------------------97",ans)
    ans_line = ans.split('answer is ')
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()
    
    if mode == 'multiple_choice':
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        for option in options:
            if option in ans:
                ans = option[1]
                break
        return ans
    elif mode == 'free_form':
        if ans[-1] == '.':
            ans = ans[:-1]
        return ans

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_numbers(string):
    return [int(num) for num in re.findall(r'\d+', string)][0]


def rate_clustering(df, seed,total_samples):
    label_counts = df['cluster_label'].value_counts(normalize=True)
    normalized_proportions = label_counts / label_counts.sum()
    sample_sizes = np.floor(normalized_proportions * total_samples).astype(int)
    sample_sizes.iloc[-1] += (total_samples - sample_sizes.sum())

    sampled_data = pd.DataFrame()
    for label in sample_sizes.index:
        sampled_df = df[df['cluster_label'] == label].sample(n=sample_sizes[label], random_state=seed)
        sampled_data = pd.concat([sampled_data, sampled_df])
    print(sampled_data['cluster_label'].value_counts())
    return sampled_data


def half_half(seed, data):

    df = data
    converted_data = df['embedding']
    array_2d = np.array(converted_data.tolist())
    print("array_2d")
    print(array_2d)


    index_pair = find_lowest_similarity_pair(array_2d)
    i_first = df.iloc[index_pair[0]]['input']
    i_second = df.iloc[index_pair[1]]['input']
    input_sentence = df['input'].tolist()
    sentence1 = input_sentence[index_pair[0]]
    sentence2 = input_sentence[index_pair[1]]
    input_sentence.remove(sentence1)
    input_sentence.remove(sentence2)
    similarity_list = array_2d
    select_list_embedding = np.array([similarity_list[index_pair[0]], similarity_list[index_pair[1]]])
    similarity_list = np.delete(similarity_list, [index_pair[0], index_pair[1]], axis=0)

    select_list = [i_first, i_second]
    number_list = 10
    num = 2

    def mean_similarity(select_embeddings, candidate_embedding):
        select_list_embedding = np.array(select_embeddings)
        candidate_embedding = np.array(candidate_embedding)
        similarity = cosine_similarity(candidate_embedding, select_list_embedding)
        average_cos_sim = np.mean(similarity)
        return average_cos_sim

    while num < number_list:
        min_sim = np.inf
        I_f = None
        candidate_embedding = None
        for index, candidate in enumerate(input_sentence):
            candidate_embedding = similarity_list[index]
            candidate_embedding = candidate_embedding.reshape(1, -1)
            sim_t = mean_similarity(select_list_embedding, candidate_embedding)
            if sim_t < min_sim:
                min_sim = sim_t
                I_f = candidate
                remove_index = index
                candidate_embedding = similarity_list[index]
        input_sentence.remove(I_f)
        select_list.append(I_f)
        similarity_list = np.delete(similarity_list, remove_index, axis=0)
        select_list_embedding = np.vstack((select_list_embedding, candidate_embedding))

        num += 1

    values_to_find = select_list

    column_name = 'input'

    filtered_df = df[df[column_name].isin(values_to_find)]

    unselected_df = df[~df[column_name].isin(values_to_find)]

    # def select_two_per_cluster(group):
    #     return group.sample(n=2) if len(group) >= 2 else group
    #
    # selected_unselected_df = unselected_df.groupby("Cluster").apply(select_two_per_cluster).reset_index(drop=True)
    selected_unselected_df = rate_clustering(unselected_df, seed, total_samples=10)
    combined_df = pd.concat([filtered_df, selected_unselected_df])
    return combined_df
    # combined_df.to_csv('half_half.csv', index=False)
    # print("half_half done")
