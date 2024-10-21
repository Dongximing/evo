import re

# Define the pattern to match the "new score" in each line
pattern = re.compile(r"new score: ([+-]?[0-9]*\.?[0-9]+)")

# Path to the text file
text_file_path = 'extracted_lines.txt'

# Initialize an empty list to store the scores
scores = []

# Read the text file and extract scores
with open(text_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            scores.append(float(match.group(1)))

# Print the list of extracted scores
def compute_avg_max(score_list):
    if not score_list:
        return (0, 0)
    average = sum(score_list) / len(score_list)
    maximum = max(score_list)
    return (average, maximum)

# Process scores in chunks of 10
results = []
for i in range(0, len(scores), 10):
    chunk = scores[i:i+10]
    avg, max_score = compute_avg_max(chunk)
    results.append((avg, max_score))

# Print results for each chunk
for idx, result in enumerate(results):
    print(f"Chunk {idx+1}: Average = {result[0]}, Max = {result[1]}")
