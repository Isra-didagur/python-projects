from math import prod
from collections import Counter
# Find the most repetitive number in the list
L1 = [1, 2, 2, 3, 2, 3, 4, 5]
def most_repetitive_number(lst):
    count = Counter(lst)
    return max(count, key=count.get)

# Example usage
most_repetitive = most_repetitive_number(L1)
print(f"The most repetitive number in the list is {most_repetitive}.")


# Calculate the product of all elements in the list
product_of_elements = prod(L1)

print(f"The product of all elements in the list is {product_of_elements}.")