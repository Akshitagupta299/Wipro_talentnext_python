#1. Write a program to remove a given item from the set.

my_set = {10, 20, 30, 40, 50}
my_set.discard(30)  
print("Set after removing 30:", my_set)






#2. Write a program to create an intersection of sets.

set1 = {10, 20, 30, 40}
set2 = {30, 40, 50, 60}
intersection = set1 & set2 
print("Intersection:", intersection)






#3. Write a program to create a union of sets.

set1 = {10, 20, 30}
set2 = {30, 40, 50}
union = set1 | set2 
print("Union:", union)






#4. Write a program to find the maximum and minimum value in a set.

my_set = {5, 10, 15, 20, 25}
print("Maximum value:", max(my_set))
print("Minimum value:", min(my_set))