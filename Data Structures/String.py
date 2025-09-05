#1. Write a program to count the number of upper and lower case letters in a String.

s = input("Enter a string: ")
upper = 0
lower = 0

for char in s:
    if char.isupper():
        upper += 1
    elif char.islower():
        lower += 1

print("Uppercase letters:", upper)
print("Lowercase letters:", lower)






#2. Write a program that will check whether a given string is palindrome or not.

s = input("Enter a string: ")
if s == s[::-1]:
    print("Palindrome")
else:
    print("Not a palindrome")






#3. Given a string, return a new string made of n copies of the first 2 chars of the original string.

s = "Wipro"
n = len(s)
first2 = s[:2]
result = first2 * n
print("Result:", result)






#4. Given a string, if the first or last character is 'x', return the string without those characters.


s = input("Enter a string: ")
if s.startswith('x'):
    s = s[1:]
if s.endswith('x'):
    s = s[:-1]
print("Modified string:", s)






#5. Given a string and an integer n, return a string made of n repetitions of the last n characters.


s = "Wipro"
n = 3
last_n = s[-n:]
result = last_n * n
print("Result:", result)