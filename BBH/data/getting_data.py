import json

# Load the dataset
task = "metaphor_boolean"
with open("/Users/ximing/Desktop/EvoPrompt/BBH/data/%s.json" % task, 'r') as file:
    data = json.load(file)["examples"]

# Modify the dataset
for index,example in enumerate(data):
    # Adding prefix and tail to the 'input'
    example["input"] = """"The essence of the task: Given a metaphoric sentence, identify if the second sentence is the correct paraphrase of the first.""" + example["input"] + "\nOptions:\n- True\n- False"

    # Changing 'target_scores' to 'target' and updating its value
    print(index)
    if example["target_scores"]["True"] == 1.0:
        example["target"] = "True"
    elif example["target_scores"]["False"] == 1.0:
        example["target"] = "False"
    # elif example["target_scores"]["contradiction"] == 1:
    #     example["target"] = "contradiction"
    del example["target_scores"]  # Remove the old key

# Optionally, you can split the dataset into training and testing sets
training_dataset = data[:200]
testing_dataset = data[200:300]

# Print the length of the dataset to verify the number of examples loaded
print(len(data))

# Optionally, save the modified dataset back to JSON
with open("/Users/ximing/Desktop/EvoPrompt/BBH/data/%s_modified.json" % task, 'w') as file:
    json.dump({"examples": data}, file, indent=4)
