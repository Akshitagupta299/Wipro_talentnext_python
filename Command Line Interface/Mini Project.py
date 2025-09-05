#Happiness

import sys

if len(sys.argv) != 4:
    print("Please provide exactly three strings as command line arguments.")
    sys.exit()

liked_str = sys.argv[1]
disliked_str = sys.argv[2]
received_str = sys.argv[3]

liked = set(liked_str.split('-'))
disliked = set(disliked_str.split('-'))
received = received_str.split('-')

happiness = 0

for num in received:
    if num in liked:
        happiness += 1
    elif num in disliked:
        happiness -= 1

print(happiness)