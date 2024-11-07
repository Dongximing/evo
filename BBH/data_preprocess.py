import pandas as pd
import json
from openai import OpenAI
#
from sklearn.cluster import KMeans
client = OpenAI(api_key='')

def get_embedding(text, model):
    text = text.replace("\n", "")
    text = text.replace("""The essence of the task: Given a metaphoric sentence, identify if the second sentence is the correct paraphrase of the first.""", "")
    print(text)
    return client.embeddings.create(input=text, model=model, dimensions=100).data[0].embedding


def generate_sentence_embedding(task):
    dataset = json.load(open("/Users/ximing/Desktop/EvoPrompt/BBH/data/%s_modified.json" % task))["examples"]
    print(len(dataset))
    training_dataset = dataset[:200]
    testing_dataset = dataset[200:300]
    for ele in training_dataset:
        ele["embedding"] = get_embedding(ele["input"], "text-embedding-3-small")
    embeddings = [example['embedding'] for example in training_dataset]
    n_clusters = 5  # Set the number of clusters, you can change this based on your needs
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)

    # Add cluster labels to each example
    for example, label in zip(training_dataset, labels):
        example['cluster_label'] = int(label)

    with open(f'/Users/ximing/Desktop/EvoPrompt/BBH/data/{task}_train_data.json', 'w') as f:
        json.dump(training_dataset, f, indent=4)
    with open(f'/Users/ximing/Desktop/EvoPrompt/BBH/data/{task}_test_data.json', 'w') as f:
        json.dump(testing_dataset, f, indent=4)


generate_sentence_embedding(task="metaphor_boolean")




