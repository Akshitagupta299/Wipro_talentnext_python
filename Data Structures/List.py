#1

nums = [10, 20, 30, 40, 50]
n = int(input("enter index: "))
print(nums[n])






#2. Write a program to append a new item to the end of the list.

nums = [1, 2, 3, 4]
n = int(input("enter element: "))
nums.append(n)
print(nums)






#3. Write a program to reverse the order of the items in the list.

nums = [1, 2, 3, 4, 5]
nums.reverse()
print(nums)






#4. Write a program to print the number of occurrences of a specified element in a list.

nums = [1, 2, 3, 2, 4, 2, 5]
n = int(input("enter no: "))
count = nums.count(n)
print(f"Number of times {n} occurs: ", count)






#5. Write a program to append the items of list1 to list2 in the front.

list1 = [10, 20]
list2 = [30, 40]
list2 = list2 + list1
print("Combined list:", list2)






#6. Write a program to insert a new item before the second element in an existing list.

nums = [1, 3, 4, 5]
n = int(input("enter element: "))
nums.insert(1, n) 
print(nums)






#7. Write a program to remove the item from a specified index in a list.

nums = [10, 20, 30, 40, 50]
index_to_remove = int(input("enter index to remove: "))
del nums[index_to_remove]
print("updated list: ", nums)






#8. Write a program to remove the first occurrence of a specified element from a list.

nums = [5, 10, 15, 10, 20]
n = int(input("enter element to remove: "))
print(f"List after removing first occurrence of {n}: ", nums)