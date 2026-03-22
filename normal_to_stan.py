# Given data
data = [
    [8.34, 40.77, 1010.84, 90.01, 480.48],
    [23.64, 58.49, 1011.4, 74.2, 445.75]
]

# Function to standardize values
def standardize(data):
    # Flatten the data to standardize all the values at once
    flat_data = [item for sublist in data for item in sublist]
    mean = sum(flat_data) / len(flat_data)
    std = (sum((x - mean) ** 2 for x in flat_data) / len(flat_data)) ** 0.5
    standardized_data = [(x - mean) / std for x in flat_data]
    
    # Reorganize the data into the same shape as the input
    result = [standardized_data[i:i + len(data[0])] for i in range(0, len(standardized_data), len(data[0]))]
    return result

# Standardizing the given data
standardized_values = standardize(data)
standardized_values
