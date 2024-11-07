import json

# Load the dataset
task = "implicatures"
with open("/Users/ximing/Desktop/EvoPrompt/BBH/data/%s.json" % task, 'r') as file:
    data = json.load(file)["examples"]

# Modify the dataset
for example in data:
    # Adding prefix and tail to the 'input'
    example["input"] = "Does Speaker 2's answer mean yes or no? " + example["input"] + "\nOptions:\n- Yes\n- No"

    # Changing 'target_scores' to 'target' and updating its value
    if example["target_scores"]["yes"] == 1.0:
        example["target"] = "Yes"
    elif example["target_scores"]["no"] == 1.0:
        example["target"] = "No"
    del example["target_scores"]  # Remove the old key

# Optionally, you can split the dataset into training and testing sets
training_dataset = data[:200]
testing_dataset = data[200:300]

# Print the length of the dataset to verify the number of examples loaded
print(len(data))

# Optionally, save the modified dataset back to JSON
with open("/Users/ximing/Desktop/EvoPrompt/BBH/data/%s_modified.json" % task, 'w') as file:
    json.dump({"examples": data}, file, indent=4)
