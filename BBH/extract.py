import re

# Define the pattern to match lines
#pattern = re.compile(r" step (\d+), pop (\d+)    prompt (.+) ----------- new score: ([\d\.]+)")
pattern = re.compile(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] -  step (\d+), pop (\d+)\s+prompt (.+?) ----------- new score: ([+-]?[0-9]*\.?[0-9]+)")

#logger.info(f" step {step}, pop {j}    prompt {de_prompt} ----------- new score: {de_eval_res}")

# Path to the log file
log_file_path = '/Users/ximing/Desktop/EvoPrompt/BBH/scripts/outputs/sports_understanding/cot/de/bd8_top10_para_topk_init/turbo/sample/seed100/evol.log'
# Path to the output text file
output_file_path = 'extracted_lines.txt'

# Read the log file and extract matching lines
with open(log_file_path, 'r') as file:
    lines = file.readlines()
print(lines)
# Filter lines that match the pattern
extracted_lines = [line for line in lines if pattern.match(line)]

# Write the extracted lines to the output text file
with open(output_file_path, 'w') as file:
    file.writelines(extracted_lines)

print(f"Extracted lines are saved in {output_file_path}")
