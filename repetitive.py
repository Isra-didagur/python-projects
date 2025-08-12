import random

L1 = [1, 2, 2, 3, 2, 3, 4, 5]

# Randomly sample elements from the list multiple times
samplesize=1000
samples = [random.choice (L1) for i in range (samplesize)]

# Count occurrences in the sampled data
most_frequent =max(set(samples), key=samples.count)

print(f"The most repetitive element (estimated) is {most_frequent}.")

