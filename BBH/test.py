import numpy as np

# Example function that returns a 20x5 array
def example_function():
    return np.random.rand(20, 5)  # Replace with your actual function

# Initialize an empty list to store results
results = []

# Call the function multiple times and append the results
for _ in range(3):  # Replace 3 with the number of times you call the function
    array = example_function()
    print(array)
    results.append(array)

# Concatenate all 20x5 arrays horizontally
final_result = np.hstack(results)

# Display the result
print("Final combined array shape:", final_result)
