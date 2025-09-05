#1. program to check if given no. is positive, negetive or zero

def check_number():
    n = int(input("enter no."))
    if n > 0:
        return "positive"
    if n <0:
        return "negetive"
    if n == 0:
        return "zero"
print(check_number())







#2. program to check if given no. is even or odd

def check_number():
    n = int(input("enter no."))
    if n%2 == 0:
        return "even"
    else:
        return "odd"
print(check_number())






#3. check is last digit same or not

def check_last_digit():
    n1 = int(input("enter no."))
    n2 = int(input("enter no."))
    a = list(str(n1))
    b = list(str(n2))
    if a[-1] == b[-1]:
        return True
    else:
        return False
print(check_last_digit())






#4. print no's from 1 to 10 in single row with one tab space

def print_numbers():
    for i in range(1,11):
        print(i,end=' ')
print_numbers()






#5. print even no's between 23 and 57 each no. in seprate row

def print_even_numbers():
    for i in range(23,57):
        if i%2 == 0:
            print(i)
print_even_numbers()






#6. check if given no. is prime or not

def prime_number():
    n = int(input("enter no."))
    for i in range(2, int(n**0.5) + 1):  
        if n % i == 0:
            print("Not a prime number.")
            return
    print("Prime number.")
prime_number()






#7. program to print prime no. between 10  and 99

def  prime_numbers():
    for i in range(10,99):
        is_prime = True
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            print(i)

prime_numbers()






#8. print sum of all digit in given number

def sum_digit():
    n = int(input("enter no."))
    total = 0
    for i in str(n):
        total += int(i)
    print(total)
sum_digit()






#9. reverse the given number and print

def reverse_number():
    n = int(input("enter no."))
    a = str(n)
    result = a[::-1]
    print(int(result))

reverse_number()






#10. find if a given no. is palindrome of not

def check_palindrome():
    n = int(input("Enter a number: "))
    a = str(n)
    result = a[::-1]
    if a == result:
        return "Palindrome"
    else:
        return "Not a palindrome"

print(check_palindrome())