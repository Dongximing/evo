import json

# Load the dataset
task = "nli"
with open("/Users/ximing/Desktop/EvoPrompt/BBH/data/%s.json" % task, 'r') as file:
    data = json.load(file)["examples"]

# Modify the dataset
for index,example in enumerate(data):
    # Adding prefix and tail to the 'input'
    example["input"] = """This is a natural language inference task. There are two sentences in English. The answer is "entailment" if the first sentence entails the second, "contradiction" if the second sentence contradicts the first, and "neutral" if neither is of those two cases holds.""" + example["input"] + """ 
Options:
- entailment
- neutral
- contradiction"""

    # Changing 'target_scores' to 'target' and updating its value
    print(index)
    if example["target_scores"]["entailment"] == 1:
        example["target"] = "ent"
    elif example["target_scores"]["neutral"] == 1:
        example["target"] = "neutral"
    elif example["target_scores"]["contradiction"] == 1:
        example["target"] = "contr"
    del example["target_scores"]  # Remove the old key

# Optionally, you can split the dataset into training and testing sets
training_dataset = data[:200]
testing_dataset = data[200:300]
with open(f'/Users/ximing/Desktop/EvoPrompt/BBH/data/{task}_train_data.json', 'w') as f:
    json.dump(training_dataset, f, indent=4)
with open(f'/Users/ximing/Desktop/EvoPrompt/BBH/data/{task}_test_data.json', 'w') as f:
    json.dump(testing_dataset, f, indent=4)

# Print the length of the dataset to verify the number of examples loaded
print(len(data))

# Optionally, save the modified dataset back to JSON
with open("/Users/ximing/Desktop/EvoPrompt/BBH/data/%s_modified.json" % task, 'w') as file:
    json.dump({"examples": data}, file, indent=4)
