#1. Write a program to accept two numbers as command line arguments and display their sum.

import sys

if len(sys.argv) != 3:
    print("Please provide exactly two numbers as command line arguments.")
else:
    try:
        num1 = int(sys.argv[1])
        num2 = int(sys.argv[2])
        print("Sum:", num1 + num2)
    except ValueError:
        print("Please provide valid integer numbers.")






#2

import sys

if len(sys.argv) < 2:
    print("Please provide a welcome message.")
else:
    script_name = sys.argv[0]
    message = " ".join(sys.argv[1:])
    print("Script Name:", script_name)
    print("Message:", message)






#3

import sys

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

if len(sys.argv) != 11:
    print("Please provide exactly 10 numbers as command line arguments.")
else:
    try:
        numbers = list(map(int, sys.argv[1:]))
        prime_sum = sum(n for n in numbers if is_prime(n))
        print("Sum of prime numbers:", prime_sum)
    except ValueError:
        print("Please provide only integer values.")