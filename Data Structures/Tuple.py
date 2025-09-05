#1. Write a program to print the 4th element from first and 4th element from last in a tuple.

t = (5, 10, 15, 20, 25, 30, 35, 40)
print("4th element from start:", t[3])      
print("4th element from end:", t[-4])






#2. Write a program to check whether an element exists in a tuple or not.

t = (1, 3, 5, 7, 9)
element = int(input("Enter a number to check: "))

if element in t:
    print("Element exists in the tuple.")
else:
    print("Element does not exist.")






#3. Write a program to convert a list into a tuple.

my_list = [10, 20, 30, 40]
my_tuple = tuple(my_list)
print("Converted tuple:", my_tuple)






#4. Write a program to find the index of an item in a tuple.

t = ('apple', 'banana', 'cherry', 'banana')
item = 'banana'
index = t.index(item)
print(f"Index of '{item}':", index)






#5. Write a program to replace last value of tuples in a list to 100.

sample_list = [(10, 20, 40), (40, 50, 60), (70, 80, 90)]
updated_list = [t[:-1] + (100,) for t in sample_list]
print("Updated list:", updated_list)
