#Sort the colors
def sort_colors(color_string):
    colors = color_string.split('-')
    sorted_colors = sorted(colors)
    return '-'.join(sorted_colors)

print(sort_colors("green-red-yellow-black-white"))    
print(sort_colors("PINK-BLUE-TAN-PURPLE"))     
       







#Playing with names
def ispalindrome(name):
    return name == name[::-1]

def count_the_vowels(name):
    vowels = "aeiouAEIOU"
    return sum(1 for char in name if char in vowels)

def frequency_of_letters(name):
    freq = {}
    for char in name:
        freq[char] = freq.get(char, 0) + 1
    return freq


name = input("Enter a name: ")
if ispalindrome(name):
    print("Yes it is a palindrome.")
else:
    print("No it is not a palindrome.")

vowel_count = count_the_vowels(name)
print(f"No of vowels: {vowel_count}")


freq = frequency_of_letters(name)
print("Frequency of letters:", end=" ")
print(', '.join(f"{char}-{count}" for char, count in freq.items()))