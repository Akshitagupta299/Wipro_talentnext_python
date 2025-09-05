#1. Function to return the sum of all numbers in a list

def sum_of_list(numbers):
    return sum(numbers)

print(sum_of_list([8, 2, 3, 0, 7])) 






#2. Function to return the reverse of a string

def reverse_string(s):
    return s[::-1]

print(reverse_string("1234abcd")) 






#3. Function to calculate the factorial of a number

def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  






#4. Function to count uppercase and lowercase letters in a string

def count_case_letters(s):
    upper = sum(1 for c in s if c.isupper())
    lower = sum(1 for c in s if c.islower())
    print("Uppercase letters:", upper)
    print("Lowercase letters:", lower)

count_case_letters("Hello World")  






#5. Function to return even numbers from a given list

def even_numbers(lst):
    return [num for num in lst if num % 2 == 0]

print(even_numbers([1, 2, 3, 4, 5, 6, 7, 8, 9]))  






#6. Function to check if a number is prime

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

print(is_prime(7)) 