import re

# Define the pattern to match the "new score" along with the full prompt in each line
pattern = re.compile(r"prompt\s+(.+?)\s*-----------\s*new score:\s*([+-]?[0-9]*\.?[0-9]+)")

# Path to the text file
text_file_path = 'extracted_lines.txt'

# Initialize a list to store the prompts and scores as tuples
prompts_scores = []

# Read the text file and extract prompts and scores
with open(text_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            prompt = match.group(1)
            score = float(match.group(2))
            prompts_scores.append((prompt, score))

# Function to find the prompt with the highest score
def find_max_prompt(prompts_scores):
    if not prompts_scores:
        return ("", 0)
    # Find the tuple with the max score
    max_tuple = max(prompts_scores, key=lambda item: item[1])
    return max_tuple

# Process scores in chunks of 10 and find the max prompt for each chunk
results = []
for i in range(0, len(prompts_scores), 10):
    chunk = prompts_scores[i:i+10]
    prompt, max_score = find_max_prompt(chunk)
    results.append((prompt, max_score))
sentence_list = []
# Print results for each chunk
for idx, result in enumerate(results):
    print(f"Chunk {idx+1}: Prompt with highest score = '{result[0]}', Score = {result[1]}")
    sentence_list.append(result[0])
import editdistance

def calculate_edit_distances(sentences):
    distances = []

    for i in range(len(sentences) - 1):
        dist = editdistance.eval(sentences[i], sentences[i + 1])
        distances.append((sentences[i], sentences[i + 1], dist))
    return distances
print(calculate_edit_distances(sentence_list))
