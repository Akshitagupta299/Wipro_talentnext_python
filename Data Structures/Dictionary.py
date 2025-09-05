#1. Write a program to add a key and value to a dictionary.

d = {0: 10, 1: 20}
d[2] = 30
print("Updated dictionary: ", d)






#2. Write a program to concatenate the following dictionaries to create a new one.

dic1 = {1: 10, 2: 20}
dic2 = {3: 30, 4: 40}
dic3 = {5: 50, 6: 60}
combined = {}

for d in (dic1, dic2, dic3):
    combined.update(d)

print("Combined dictionary:", combined)






#3. Write a program to check if a given key already exists in a dictionary.

d = {1: "apple", 2: "banana", 3: "cherry"}
key = int(input("Enter a key to check: "))

if key in d:
    print("Key exists.")
else:
    print("Key does not exist.")

'''4. Write a program to iterate over a dictionary using a for loop
and print keys alone, values alone, and both keys and values.'''

d = {'a': 1, 'b': 2, 'c': 3}

print("Keys:")
for key in d:
    print(key)

print("Values:")
for value in d.values():
    print(value)

print("Key-Value pairs:")
for key, value in d.items():
    print(key, ":", value)

'''5. Write a program to prepare a dictionary where the keys are numbers between 1
and 15 (both included) and the values are square of the keys.'''

squares = {}
for i in range(1, 16):
    squares[i] = i * i

print("Dictionary of squares:", squares)






#6. Write a program to sum all the values in a dictionary, considering the values are of int type.

d = {'a': 10, 'b': 20, 'c': 30}
total = sum(d.values())
print("Sum of values:", total)