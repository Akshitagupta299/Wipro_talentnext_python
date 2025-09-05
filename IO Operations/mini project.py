#  Find The Secret  ( Mini Project)

def get_meeting_details(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("File not found.")
        return


    num_lines = len(lines)
    hour = num_lines % 12
    hour = 12 if hour == 0 else hour
    meeting_time = f"{hour} AM" if num_lines <= 12 else f"{hour} PM"


    from collections import Counter
    import re

    words = []
    for line in lines:

        words += re.findall(r'\b\w+\b', line.lower())

    word_counts = Counter(words)

    most_common_word, _ = word_counts.most_common(1)[0]


    meeting_place = most_common_word.capitalize() + " Street"


    print("Meeting time:", meeting_time)
    print("Meeting place:", meeting_place)

filename = input("Enter the file name (e.g., sample.txt): ")
get_meeting_details(filename)
