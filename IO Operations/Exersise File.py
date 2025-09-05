# 1. Write a program to read the entire content from a txt file and display it to the user.

filename = input("Enter the file name: ")

with open(filename, "r") as file:
    content = file.read()

print("File Content:\n", content)







# 2. Write a program to read first n lines from a txt file. Get nas user input.

filename = input("Enter the file name: ")
n = int(input("Enter number of lines to read: "))

with open(filename, "r") as file:
    for i in range(n):
        line = file.readline()
        if not line:
            break
        print(line.strip())






# 3. Write a program to accept input from user and append it to a txt file.

filename = input("Enter the file name: ")
data = input("Enter text to append: ")

with open(filename, "a") as file:
    file.write(data + "\n")

print("Text appended successfully.")






# 4. Write a program to read contents from a txt file line by line and store each line into a list.

filename = input("Enter the file name: ")

with open(filename, "r") as file:
    lines = file.readlines()

line_list = [line.strip() for line in lines]
print("Lines stored in list:", line_list)






# 5. Write a program to find the longest word from the txt file contents, assuming that the file will have only one longest word in it.

filename = input("Enter the file name: ")

with open(filename, "r") as file:
    words = file.read().split()

longest = max(words, key=len)
print("Longest word:", longest)






# 6. Write a program to count the frequency of a user entered word in a txt file.

filename = input("Enter the file name: ")
search_word = input("Enter the word to count: ")

with open(filename, "r") as file:
    content = file.read().split()

count = content.count(search_word)
print(f"The word '{search_word}' appears {count} time(s).")
