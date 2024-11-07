import json
import os
import numpy as np
import heapq
import random
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from utils import setup_log, k_init_pop
from utils import (
    read_lines,
    get_final_prompt,
    extract_numbers,
    rate_clustering,
    half_half
)
from llm_client import paraphrase, llm_query
from data.template_ga import templates_2
from data.templates import *
from run_bbh import eval_task
import functools

import editdistance


def calculate_edit_distances(sentences):
    distances = []

    for i in range(len(sentences) - 1):
        dist = editdistance.eval(sentences[i], sentences[i + 1])
        distances.append((sentences[i], sentences[i + 1], dist))
    return distances


def calculate_edit_distances_for_all_prompt(sentences, logger):
    distances = []

    for i in range(len(sentences) - 1):
        set_a = sentences[i]
        set_b = sentences[i + 1]
        for k in set_a:
            score = 0
            for j in set_b:
                dist = editdistance.eval(k, j)
                score += dist
                distances.append((k, j, dist))
            logger.info(f"the prompt :{j} has average score {score / len(set_b)}edit distance for set:----> {set_b}")
    return distances




def dynamic_reshape(arr, new_shape, row_skip, elements_per_row):
    # Initialize the new array with zeros or an appropriate default value
    new_arr = np.zeros(new_shape, dtype=arr.dtype)

    # Get the total number of sets you need (assuming new_shape[0] matches this)
    num_sets = new_shape[0]
    # Calculate number of element groups per new row
    num_groups = new_shape[1] // elements_per_row

    # Iterate over each set
    for i in range(num_sets):
        # Extract elements based on row skip and elements per row
        for j in range(num_groups):  # This determines how many groups per row
            start_row = row_skip * j + i
            # Make sure we do not go out of original array bounds
            if start_row < arr.shape[0] and j * elements_per_row < new_shape[1]:
                # Start column is always 0 and we take the number of columns specified
                end_col = min(elements_per_row, arr.shape[1])  # Ensure not exceeding original cols
                new_arr[i, j * elements_per_row:(j + 1) * elements_per_row] = arr[start_row, :end_col]

    return new_arr


#


class Evoluter:
    def __init__(self, args, llm_config, client, sampling_method):
        self.init_poplulation = []
        self.population = []
        self.scores = []
        self.marks = []
        self.prompts2mark = {}
        self.evaluated_prompts = {}
        self.sampling_method = sampling_method

        self.client, self.llm_config = client, llm_config
        self.public_out_path = args.output
        self.task = args.task
        self.task_prompt = open("/mnt/hdd-data/shaowei/data_selection/evo/BBH/lib_prompt/%s.txt" % self.task, "r").read()

        self.logger = logger = setup_log(
            os.path.join(self.public_out_path, f"evol.log")
        )
        logger.info("=" * 50)
        logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
        logger.info("=" * 50)
        self.args = args

        self.out_path = os.path.join(self.public_out_path, f"dev_result.txt")
        self.task_data = json.load(open("/mnt/hdd-data/shaowei/data_selection/evo/BBH/data/%s.json" % args.task))["examples"]
        # self.dev_data = random.sample(self.task_data, args.sample_num)
        # print(self.dev_data)

        dev_data = json.load(open(f"/mnt/hdd-data/shaowei/data_selection/evo/BBH/data/{args.task}_train_data.json"))
        self.test_data = json.load(open(f"/mnt/hdd-data/shaowei/data_selection/evo/BBH/data/{args.task}_test_data.json"))

        if self.sampling_method == "anchor_half_sampling":
            logger.info(
                "-----there is a sampling method---------"
            )

            self.dev_data = dev_data
        elif self.sampling_method == "anchor_concatenate":
            logger.info(
                "-----concatenate method---------"
            )
            self.dev_data = dev_data
        elif self.sampling_method == "anchor_all":
            logger.info(
                "-----concatenate method---------"
            )
            self.dev_data = dev_data
        elif self.sampling_method == "sampling_dynamic":
            logger.info(
                "-----there is a sampling_dynamic method---------"
            )
            self.dev_data = random.sample(dev_data, 50)


        elif self.sampling_method == "cluster":
            logger.info(
                "-----there is a cluster method---------"
            )
            df = pd.DataFrame(dev_data)
            sampled_data = rate_clustering(df, 42, args.sample_num)
            self.dev_data = sampled_data.to_dict(orient='records')

        elif self.sampling_method == "half_half":
            logger.info(
                "-----there is a half_half method---------"
            )
            df = pd.DataFrame(dev_data)
            seed = random.randint(1,100)
            sampled_data = half_half(seed=seed, data=df)
            self.dev_data = sampled_data.to_dict(orient='records')
        elif self.sampling_method == "paper_method":
            pass
        elif self.sampling_method == "cluster_paper":
            pass
        elif self.sampling_method == "anchor":
            logger.info(
                "-----there is a anchor method---------"
            )
            self.dev_data = random.sample(dev_data, args.sample_num)
        elif self.sampling_method == "sampling":
            logger.info(
                "-----there is a anchor method---------"
            )
            self.dev_data = random.sample(dev_data, 20)
        elif self.sampling_method == "full_size_baseline":
            self.dev_data = dev_data
        else:
            logger.info(
                "-----there is a half_half method---------"
            )

            self.dev_data = random.sample(dev_data, len(dev_data))



        base2_int = functools.partial(int, base=2)
        a = base2_int("10")

        model = "turbo" if "turbo" in args.llm_type else "davinci"

        self.eval_func = functools.partial(
            eval_task,
            task=self.task,
            task_prompt=self.task_prompt,
            client=client,
            model_index=model,
            logger=logger,
            demon=args.demon,
            seed = args.seed,
            **llm_config,
        )

    def sorted(self):
        best_score = 0
        total_score = 0
        with open(os.path.join(self.public_out_path, "dev_result.txt"), "w") as wf:
            self.scores, self.population, self.marks = (
                list(t)
                for t in zip(
                *sorted(
                    zip(self.scores, self.population, self.marks),
                    key=lambda x: x[0],
                    reverse=True,
                )
            )
            )
            for score, prompt, mark in zip(self.scores, self.population, self.marks):
                float_score = float(score)
                if float_score > best_score:
                    best_score = float_score
                total_score += float_score
                wf.write(f"{mark}\t{prompt}\t{score}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {total_score / len(self.scores)}\n")
            wf.close()

    def run(self):
        self.evolute()
        self.sorted()

    def init_pop(self):
        args = self.args
        logger = self.logger

        out_path = self.public_out_path
        cur_budget = -1
        cot_cache_path = args.cot_cache_path
        desc_cache_path = args.desc_cache_path

        def load_cache(self, cache_path):
            try:
                cache = json.load(open(cache_path, "r"))
                logger.info(f"---loading prompts from {cache_path}---")
                self.evaluated_prompts = dict(
                    sorted(
                        cache.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                init_population = [k for k in list(self.evaluated_prompts.keys())]
            except:
                topk_population = []
                self.evaluated_prompts = {}
                prompt_path = (
                    f"../auto_prompts/{args.task}.txt"
                    if args.initial == "ape"
                    else "../prompts.txt"
                )
                print("prompt_path", prompt_path)
                pop = read_lines(prompt_path)
                logger.info(
                    "-----evaluating initial population and paraphrasing topk---------"
                )
                print("BBH/evoluter.py:228",len(pop))
                print("BBH/evoluter.py:229",len(self.dev_data))
                for prompt in pop:
                    eval_res, _ = self.eval_func(cot_prompt=[prompt], eval_data=self.dev_data,anchor=True,discrete= False)
                    self.evaluated_prompts[prompt] = eval_res
                    topk_population.append((eval_res, prompt))
                topk_population.sort(reverse=True, key=lambda x: x[0])

                with open(cache_path, "w") as wf:
                    self.evaluated_prompts = dict(
                        sorted(self.evaluated_prompts.items(), key=lambda item: item[1])
                    )
                    json.dump(self.evaluated_prompts, wf)
                init_population = [i[1] for i in topk_population]
            return init_population, self.evaluated_prompts

        if args.initial == "ckpt":
            init_population = []
            logger.info(f"------------load from file {args.ckpt_pop}------------")
            ckpt_pop = read_lines(args.ckpt_pop)[: args.popsize]
            for line in ckpt_pop:
                try:
                    mark, prompt, score = line.strip().split("\t")
                    score = float(score)
                except:
                    continue
                self.prompts2mark[prompt] = mark
                self.evaluated_prompts[prompt] = score
                init_population.append(prompt)
                cur_budget = extract_numbers(args.ckpt_pop.split("/")[-1])
            logger.info("current budget: %d" % cur_budget)
        elif args.initial == "cot":
            init_population, self.evaluated_prompts = load_cache(self, cot_cache_path)
            self.prompts2mark = {i: "manual" for i in init_population}
        elif args.initial == "desc":
            init_population, self.evaluated_prompts = load_cache(self, desc_cache_path)
            self.prompts2mark = {i: "ape" for i in init_population}

        elif args.initial == "all":
            init_population_cot, self.evaluated_prompts_cot = load_cache(
                self, cot_cache_path
            )
            init_population_desc, self.evaluated_prompts_desc = load_cache(
                self, desc_cache_path
            )
            self.evaluated_prompts = {
                **self.evaluated_prompts_cot,
                **self.evaluated_prompts_desc,
            }
            self.evaluated_prompts = dict(
                sorted(
                    self.evaluated_prompts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            init_population = [k for k in list(self.evaluated_prompts.keys())]
            self.prompts2mark = {
                i: "manual" if i in init_population_cot else "ape"
                for i in init_population
            }



        if args.initial_mode in ["para_topk", "para_bottomk", "para_randomk"]:
            k_pop = k_init_pop(args.initial_mode, init_population, k=args.popsize)
            para_population = paraphrase(
                client=self.client,
                sentence=k_pop,
                type=args.llm_type,
                temperature=0.5,
                **self.llm_config,
            )
            for i in para_population:
                self.prompts2mark[i] = "para"
            init_population = k_pop + para_population
            init_population = init_population[: args.popsize]
        elif args.initial_mode in ["topk", "bottomk", "randomk"]:
            init_population = k_init_pop(
                args.initial_mode, init_population, k=args.popsize
            )
        cur_best_score = 0
        cur_best_prompt = ""
        total_score = 0

        self.population = [i for i in init_population]
        assert len(self.population) == args.popsize

        with open(os.path.join(out_path, "step0_pop_para.txt"), "w") as wf:
            for i in self.population:
                if i not in self.evaluated_prompts:
                    init_scores, _ = self.eval_func(cot_prompt=[i], eval_data=self.dev_data,anchor=True,discrete=False)
                    self.evaluated_prompts[i] = init_scores
                scores = self.evaluated_prompts[i]
                total_score += scores
                if cur_best_score < scores:
                    cur_best_score = scores
                    cur_best_prompt = i
                wf.write(f"{self.prompts2mark[i]}\t{i}\t{scores}\n")
            wf.write(f"best score: {cur_best_score}\n")
            wf.write(f"average score: {total_score / args.popsize}\n")

        return self.evaluated_prompts, cur_budget

    def write_step(self, i, avg_score, best_score):
        out_path = self.public_out_path
        with open(os.path.join(out_path, f"step{i}_pop.txt"), "w") as wf:
            for p in self.population:
                score = self.evaluated_prompts[p]
                wf.write(f"{self.prompts2mark[p]}\t{p}\t{score}\n")
            wf.write(f"best score: {best_score}\n")
            wf.write(f"average score: {avg_score}\n")

    def evolute(self):
        raise NotImplementedError

    def calculate_anchor_point(self, populations):
        result, score = self.eval_func(cot_prompt=populations, eval_data=self.dev_data,anchor=True,discrete=False)
        print(score)
        print(type(score))
        a, b = score.shape
        c = a*b//len(self.dev_data)
        accumulated_array = dynamic_reshape(score,(len(self.dev_data),c),len(self.dev_data),b)
        return accumulated_array

    def getting_dataset_from_anchor_point(self):
        pass

    def test(self, step):
        self.logger.info(f"----------testing step {step} population----------")
        pop_marks = [self.prompts2mark[i] for i in self.population]
        pop_scores = [self.evaluated_prompts[i] for i in self.population]
        self.population, pop_scores, pop_marks = (
            list(t)
            for t in zip(
            *sorted(
                zip(self.population, pop_scores, pop_marks),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        )

        test_prompt_num = self.args.popsize // 2

        with open(
                os.path.join(self.public_out_path, f"step{step}_pop_test.txt"), "w"
        ) as wf:
            self.logger.info(f"test_prompt---------------------final in the self.population--{self.population}")
            self.logger.info(f"test_prompt---------------------final in the self.population--{pop_marks}")
            self.logger.info(f"test_prompt---------------------final in the self.population--{pop_scores}")
            evoluted = False
            manual = False
            para = False
            n = 0
            for i in tqdm(range(len(self.population))):
                test_prompt = self.population[i]
                print(f"test_prompt---------------------final --{test_prompt}")
                self.logger.info(f"test_prompt---------------------final --{test_prompt}")
                if self.prompts2mark[test_prompt] == 'evoluted' and evoluted == True:
                    if best == pop_scores[i]:
                        test_mark = pop_marks[i]
                        test_score, _ = self.eval_func(
                            cot_prompt=test_prompt, eval_data=self.test_data, anchor=False, discrete=False
                        )
                        dev_score = self.evaluated_prompts[test_prompt]
                        all_score = (
                                            test_score * len(self.test_data)
                                            + len(self.dev_data) * self.evaluated_prompts[test_prompt]
                                    ) / len(self.task_data)
                        wf.write(
                            f"{test_mark}\t{test_prompt}\t{dev_score}\t{test_score}\t{all_score}\t{self.sampling_method}\n"
                        )
                        wf.flush()

                if self.prompts2mark[test_prompt] == 'evoluted' and evoluted == False:
                    test_mark = pop_marks[i]
                    test_score, _ = self.eval_func(
                        cot_prompt=test_prompt, eval_data=self.test_data, anchor = False, discrete=False
                    )
                    dev_score = self.evaluated_prompts[test_prompt]
                    all_score = (
                                        test_score * len(self.test_data)
                                        + len(self.dev_data) * self.evaluated_prompts[test_prompt]
                                ) / len(self.task_data)
                    wf.write(
                        f"{test_mark}\t{test_prompt}\t{dev_score}\t{test_score}\t{all_score}\t{self.sampling_method}\n"
                    )
                    wf.flush()
                    evoluted = True
                    best = pop_scores[i]
                    n += 1
                if self.prompts2mark[test_prompt] == 'manual' and manual == False:
                    test_mark = pop_marks[i]
                    test_score, _ = self.eval_func(
                        cot_prompt=test_prompt, eval_data=self.test_data,anchor = False,discrete=False
                    )
                    dev_score = self.evaluated_prompts[test_prompt]
                    all_score = (
                                        test_score * len(self.test_data)
                                        + len(self.dev_data) * self.evaluated_prompts[test_prompt]
                                ) / len(self.task_data)
                    wf.write(
                        f"{test_mark}\t{test_prompt}\t{dev_score}\t{test_score}\t{all_score}\t{self.sampling_method}\n"
                    )
                    wf.flush()
                    n += 1
                    manual = True
                if self.prompts2mark[test_prompt] == 'para' and para == False:
                    test_mark = pop_marks[i]
                    test_score, _ = self.eval_func(
                        cot_prompt=test_prompt, eval_data=self.test_data, anchor = False,discrete=False
                    )
                    dev_score = self.evaluated_prompts[test_prompt]
                    all_score = (
                                        test_score * len(self.test_data)
                                        + len(self.dev_data) * self.evaluated_prompts[test_prompt]
                                ) / len(self.task_data)
                    wf.write(
                        f"{test_mark}\t{test_prompt}\t{dev_score}\t{test_score}\t{all_score}\t{self.sampling_method}\n"
                    )
                    wf.flush()
                    n += 1
                    para = True


class GAEvoluter(Evoluter):
    def __init__(self, args, llm_config, client, sampling_method):
        super(GAEvoluter, self).__init__(args, llm_config=llm_config, client=client, sampling_method=sampling_method)
        self.template = templates_2["sim"]


    def evolute(self):
        logger = self.logger
        args = self.args
        self.evaluated_prompts, cur_budget = self.init_pop()
        out_path = self.public_out_path
        template = self.template

        best_scores = []
        avg_scores = []

        logger.info(f"init  self.evaluated_prompts{self.evaluated_prompts}")
        logger.info(f"cur_budget------------------->{cur_budget}")
        logger.info(f"cur_budget------------------->{args.budget}")
        logger.info(f"init  self.population{self.population}")
        total_candidate = []
        the_best_ones = []
        find_max = False
        step = -1

        for step in range(cur_budget + 1, args.budget):

            if step == 0 and self.sampling_method.startswith("anchor"):
                dev_data = json.load(open(f"/mnt/hdd-data/shaowei/data_selection/evo/BBH/data/{args.task}_train_data.json"))
                self.dev_data = dev_data
                if os.path.exists("array_fle.npy"):
                    training_score = np.load("array_fle.npy")
                else:
                    training_score = self.calculate_anchor_point(self.population)
                    np.save('array_file.npy', training_score)
                if self.sampling_method == "anchor_concatenate":
                    embeddings = [item['embedding'] for item in dev_data]
                    embeddings_array = np.array(embeddings)
                    training_score = np.concatenate((embeddings_array,training_score), axis=1)
                logger.info(f"training_score----------->{training_score.shape}")
                filtered_data = training_score
                logger.info("**" * 50)
                trials = 5
                random_seed = 10
                anchor_point = 20
                # logger.info(f"valid_columns--------->{valid_columns}")
                # filtered_data = [self.dev_data[i] for i in range(len(self.dev_data)) if i not in valid_columns]
                # total_list = [item['input'] for item in filtered_data]
                # logger.info(f"all filtered_data {total_list}")
                # if len(filtered_data) > 20:
                #     selected_data = random.sample(filtered_data, 20)
                # else:
                #     selected_data = filtered_data
                # self.dev_data = selected_data
                # input_list = [item['input'] for item in selected_data]
                # input_list = ['Is the following sentence plausible? "Carles Puyol did a maradona on the defender."',
                #  'Is the following sentence plausible? "Patrice Bergeron converted the first down."',
                #  'Is the following sentence plausible? "Gerrit Cole committed a handball in the European Cup."',
                #  'Is the following sentence plausible? "Mitchell Marner nutmegged the defender."',
                #  'Is the following sentence plausible? "Elias Lindholm beat the buzzer."',
                #  'Is the following sentence plausible? "Nazem Kadri took a charge in the NBA Championship."',
                #  'Is the following sentence plausible? "Igor Shesterkin launched a hail mary."',
                #  'Is the following sentence plausible? "Steven Stamkos hit the slant pass."',
                #  'Is the following sentence plausible? "Philip Rivers drove into the restricted area."',
                #  'Is the following sentence plausible? "Eden Hazard hit the buzzer beater."',
                #  'Is the following sentence plausible? "Jonathan Marchessault scored on the power play in the Stanley Cup."',
                #  'Is the following sentence plausible? "Willian killed the powerplay."',
                #  'Is the following sentence plausible? "James Karinchak crossed the blue line."',
                #  'Is the following sentence plausible? "Giorgio Chiellini committed a handball in the FA Cup."',
                #  'Is the following sentence plausible? "Mookie Betts skated behind the net."',
                #  'Is the following sentence plausible? "Mark Stone spent time in the penalty box in the Stanley Cup."',
                #  'Is the following sentence plausible? "Timo Meier nutmegged the defender in the FA Cup."',
                #  'Is the following sentence plausible? "Petr Cech bricked the three pointer."',
                #  'Is the following sentence plausible? "Sam Darnold scored on the power play in the Stanley Cup."',
                #  'Is the following sentence plausible? "T.Y. Hilton threw a touchdown in the AFC divisional round."']
                # logger.info(f"self.dev_data in 185  {input_list}")
                if self.sampling_method.startswith("anchor_half_half"):
                    anchor_point =10


                kmeans_models = [
                    KMeans(n_clusters=anchor_point, random_state=1000 * t + random_seed, n_init="auto").fit(
                        filtered_data) for t in range(trials)]
                kmeans = kmeans_models[np.argmin([m.inertia_ for m in kmeans_models])]

                # Calculating anchor points
                anchor_points = pairwise_distances(kmeans.cluster_centers_, filtered_data, metric='euclidean').argmin(
                    axis=1)

                logger.info(f" anchor_points ---------------------->{anchor_points}")


                self.dev_data = [self.dev_data[i] for i in anchor_points]

                if self.sampling_method.startswith("anchor_half_half"):

                    all_data = json.load(open(f"/mnt/hdd-data/shaowei/data_selection/evo/BBH/data/{args.task}_train_data.json"))
                    remaining_data = [sample for sample in all_data if sample not in self.dev_data]
                    labels = set(sample['cluster_label'] for sample in remaining_data)
                    new_samples = []
                    for label in labels:
                        label_samples = [sample for sample in remaining_data if sample['cluster_label'] == label]
                        new_samples.extend(random.sample(label_samples, min(2, len(label_samples))))
                    self.dev_data.extend(new_samples)

                self.evaluated_prompts = {}
                for i in self.population:
                    de_eval_res, _ = self.eval_func(cot_prompt=[i], eval_data=self.dev_data, anchor=True,discrete=False)
                    self.evaluated_prompts[i] = de_eval_res
            if step == 0 and self.sampling_method.startswith("sampling_dynamic"):
                training_score = self.calculate_anchor_point(self.population)
                logger.info(f"training_score----------->{training_score.shape}")
                filtered_data = training_score
                logger.info("**" * 50)
                trials = 5
                random_seed = 10
                anchor_point = 20
                kmeans_models = [
                    KMeans(n_clusters=anchor_point, random_state=1000 * t + random_seed, n_init="auto").fit(
                        filtered_data) for t in range(trials)]
                kmeans = kmeans_models[np.argmin([m.inertia_ for m in kmeans_models])]

                # Calculating anchor points
                anchor_points = pairwise_distances(kmeans.cluster_centers_, filtered_data, metric='euclidean').argmin(
                    axis=1)

                logger.info(f" anchor_points ---------------------->{anchor_points}")

                self.dev_data = [self.dev_data[i] for i in anchor_points]
                self.evaluated_prompts = {}
                for i in self.population:
                    de_eval_res, _ = self.eval_func(cot_prompt=[i], eval_data=self.dev_data, anchor=True,
                                                    discrete=False)
                    self.evaluated_prompts[i] = de_eval_res



            total_score = 0
            best_score = 0
            logger.info("++" * 50)
            logger.info(f"615 ----------->{self.evaluated_prompts}")

            fitness = np.array([self.evaluated_prompts[i] for i in self.population])

            logger.info(f"fitness IN 546-------------------------------->{fitness}")
            input_list = [item['input'] for item in self.dev_data]
            logger.info("**" * 50)
            logger.info(f"input_list      ---------->  {input_list}")
            logger.info("**" * 50)
            new_pop = []
            if args.sel_mode == "wheel":
                wheel_idx = np.random.choice(
                    np.arange(args.popsize),
                    size=10,
                    replace=True,
                    p=fitness / fitness.sum(),
                ).tolist()  # temp self.population to select parents
                logger.info(f"wheel_idx   wheel_idx  wheel_idx {wheel_idx}")
                parent_pop = [self.population[i] for i in wheel_idx]
                logger.info(f"parent_pop   parent_pop  parent_pop {parent_pop}")
                logger.info("This is related to sampling method")
            elif args.sel_mode in ["random", "tour"]:
                parent_pop = [i for i in self.population]
            separate_candidate = []

            for j in range(args.popsize):
                logger.info("step {i}, pop {j}".format(i=step, j=j))
                if args.sel_mode in ["random", "wheel"]:
                    parents = random.sample(parent_pop, 2)
                    cand_a = parents[0]
                    cand_b = parents[1]
                elif args.sel_mode == "tour":
                    group_a = random.sample(parent_pop, 2)
                    group_b = random.sample(parent_pop, 2)
                    cand_a = max(group_a, key=lambda x: self.evaluated_prompts[x])
                    cand_b = max(group_b, key=lambda x: self.evaluated_prompts[x])

                request_content = template.replace("<prompt1>", cand_a).replace(
                    "<prompt2>", cand_b
                )
                logger.info("evolution example:")
                logger.info(request_content)
                logger.info("parents:")
                logger.info(cand_a)
                logger.info(cand_b)
                child_prompt = llm_query(
                    client=self.client,
                    data=request_content,
                    type=args.llm_type,
                    task=False,
                    temperature=0.7,
                    **self.llm_config,
                )

                child_prompt = get_final_prompt(child_prompt)
                logger.info(f"step {step}, pop {j} original child prompt: {child_prompt}")
                separate_candidate.append(child_prompt)
                de_eval_res, _ = self.eval_func(cot_prompt=[child_prompt], eval_data=self.dev_data,anchor=True,discrete=False)
                logger.info(f" step {step}, pop {j}    prompt {child_prompt} ----------- new score: {de_eval_res}")
                self.prompts2mark[child_prompt] = "evoluted"

                self.evaluated_prompts[child_prompt] = de_eval_res
                if args.ga_mode == "std":
                    selected_prompt = child_prompt
                    selected_score = de_eval_res
                    self.population[j] = selected_prompt

                elif args.ga_mode == "topk":
                    selected_prompt = child_prompt
                    selected_score = de_eval_res

                new_pop.append(selected_prompt)
                total_score += selected_score
                if selected_score > best_score:
                    best_score = selected_score
                    best_prompt = selected_prompt
                if de_eval_res == 1.0:
                    find_max = True
                    break
            total_candidate.append(separate_candidate)
            the_best_ones.append(best_prompt)
            logger.info(f"average score for is {total_score / len(new_pop)}")

            if args.ga_mode == "topk":
                double_pop = list(set(self.population + new_pop))

                logger.info("++" * 50)
                logger.info(f"517 ----------->{self.evaluated_prompts}")

                double_pop = sorted(
                    double_pop, key=lambda x: self.evaluated_prompts[x], reverse=True
                )
                logger.info(f"517 ----------->{double_pop}")
                self.population = double_pop[: args.popsize]
                total_score = sum([self.evaluated_prompts[i] for i in self.population])
                best_score = self.evaluated_prompts[self.population[0]]
            avg_score = total_score / args.popsize
            avg_scores.append(avg_score)
            best_scores.append(best_score)

            self.write_step(i=step, best_score=best_score, avg_score=avg_score)
            if find_max:
                break

        self.test(step=step)
        best_edit_distance = calculate_edit_distances(the_best_ones)
        logger.info("*" * 50)
        logger.info(f"best prompt edit distance{best_edit_distance}")
        all_edit_distance = calculate_edit_distances_for_all_prompt(total_candidate, logger)
        logger.info(f"all_edit_distance edit distance{all_edit_distance}")
        logger.info("*" * 50)
        logger.info(f"best_scores  ------->{best_scores}")

        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        logger.info(f"best_scores: {','.join(best_scores)}")
        logger.info(f"avg_scores: {','.join(avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        logger.info(f"self.scores  ------->{self.scores}")
        logger.info(f"self.marks   ------->{self.marks}")
        self.sorted()


class ParaEvoluter(Evoluter):
    def __init__(self, args, llm_config, client):
        super(ParaEvoluter, self).__init__(args, llm_config=llm_config, client=client)

    def init_pop(self):
        args = self.args
        logger = self.logger
        task = args.task
        init_prompt_path = f"./auto_prompts/{task}.txt"
        self.init_population = read_lines(init_prompt_path)[: args.popsize]
        self.prompts2mark = {i: "ape" for i in self.init_population}
        logger.info("initial population:")
        for i in self.init_population:
            logger.info(i)
        with open(f"{self.public_out_path}/init.txt", "w") as wf:
            for i in self.population:
                logger.info(i)
                wf.write(f"{i}\n")

    def evolute(self):
        self.init_pop()
        args = self.args
        k = args.popsize
        logger = self.logger
        self.evaluated_prompts = {}
        cur_budget = -1
        topk_heap = []
        best_scores, avg_scores = [], []

        if args.initial == "ckpt":
            self.init_population = []
            logger.info("cur budget is {}".format(cur_budget))
            logger.info(f"------------load from file {args.ckpt_pop}------------")
            ckpt_pop = read_lines(args.ckpt_pop)
            for line in ckpt_pop:
                try:
                    elements = line.split("\t")
                    mark, prompt = elements[0], elements[1]
                    score = elements[2:]
                except:
                    continue
                self.prompts2mark[prompt] = mark
                mean_score = float(score)
                self.evaluated_prompts[prompt] = score
                self.init_population.append(prompt)
                heapq.heappush(topk_heap, (mean_score, prompt))

                logger.info(f"{prompt}, {self.evaluated_prompts[prompt]}")
            cur_budget = extract_numbers(args.ckpt_pop.split("/")[-1])

        _ = paraphrase(
            sentence=self.init_population[0],
            client=self.client,
            type="davinci",
            **self.llm_config,
        )
        # initial population evaluation
        if args.initial != "ckpt":
            for i, prompt in enumerate(self.init_population):
                score, _ = self.eval_func(cot_prompt=[prompt],anchor=True,discrete=False)
                self.evaluated_prompts[prompt] = score
                self.logger.info(f"{self.prompts2mark[prompt]}: {prompt}, {score}")
                heapq.heappush(topk_heap, (score, prompt))

        for step in range(cur_budget + 1, args.budget):
            best_score = 0
            total_score = 0
            self.population, self.marks, self.scores = [], [], []
            self.logger.info(f"step: {step}")
            top_k = heapq.nlargest(k, topk_heap)

            new_prompts = []
            paraphrased_prompts = paraphrase(
                sentence=[i[1] for i in top_k],
                client=self.client,
                type=args.llm_type,
                temperature=0.5,
                **self.llm_config,
            )
            for i, prompt in enumerate(paraphrased_prompts):
                self.logger.info(f"step: {step}, prompt: {prompt}")
                new_score, _ = self.eval_func(cot_prompt=[prompt],anchor=True,discrete=False)
                self.prompts2mark[prompt] = "para"
                self.logger.info(f"paraphrased: {prompt}, {new_score}")
                self.logger.info(
                    f"original: {top_k[i][1]}, {self.evaluated_prompts[top_k[i][1]]}"
                )
                new_prompts.append((new_score, prompt))
                self.evaluated_prompts[prompt] = new_score
            for new_prompt in new_prompts:
                heapq.heappushpop(topk_heap, new_prompt)

            for _, prompt in topk_heap:
                self.population.append(prompt)
                cur_score = float(self.evaluated_prompts[prompt])
                if best_score < cur_score:
                    best_score = cur_score
                total_score += cur_score
                mark = "manual" if prompt in self.init_population else "para"
                self.marks.append(mark)
            avg_score = total_score / len(topk_heap)
            best_scores.append(best_score)
            avg_scores.append(avg_score)

            self.write_step(i=step, best_score=best_score, avg_score=avg_score)

            if step == args.budget - 1:
                self.test(step=step)

        best_scores = [str(i) for i in best_scores]
        avg_scores = [str(round(i, 4)) for i in avg_scores]
        logger.info(f"best_scores: {','.join(best_scores)}")
        logger.info(f"avg_scores: {','.join(avg_scores)}")
        self.scores = [self.evaluated_prompts[i] for i in self.population]
        self.marks = [self.prompts2mark[i] for i in self.population]
        self.sorted()
