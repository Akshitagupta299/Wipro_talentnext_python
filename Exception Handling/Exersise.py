# 1. Write a program to accept two numbers from the user and perform division. If any exception occurs, print an error message or else print the result.

try:
    a = int(input("Enter first number: "))
    b = int(input("Enter second number: "))
    result = a / b
    print("Result:", result)
except Exception as e:
    print("Error:", e)






# 2. Write a program to accept a number from the user and check whether it's prime or not. If user enters anything other than number, handle the exception and print an error message.

try:
    num = int(input("Enter a number: "))
    if num < 2:
        print(num, "is not a prime number.")
    else:
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                print(num, "is not a prime number.")
                break
        else:
            print(num, "is a prime number.")
except ValueError:
    print("Error: Invalid input! Please enter a valid integer.")






# 3. Write a program to accept the file name to be opened from the user, if file exist print the contents of the file in title case or else handle the exception and print an error message.

filename = input("Enter the file name: ")

try:
    with open(filename, "r") as file:
        content = file.read()
        print(content.title())
except FileNotFoundError:
    print("Error: File not found.")






# 4. Declare a list with 10 integers and ask the user to enter an index. Check whether the number in that index is positive or negative number. If any invalid index is entered, handle the exception and print an error message.

numbers = [5, -2, 10, -8, 15, -6, 3, -1, 7, -9]

try:
    index = int(input("Enter an index (0 to 9): "))
    value = numbers[index]
    if value >= 0:
        print("The number is positive.")
    else:
        print("The number is negative.")
except IndexError:
    print("Error: Invalid index! Please enter between 0 and 9.")
except ValueError:
    print("Error: Please enter a valid integer.")
