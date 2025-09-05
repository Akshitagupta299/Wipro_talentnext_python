# 1. 

input_list = [1, 2, 3, 4, 5, 6, 7]
output_dict = {x: x**3 for x in input_list if x % 2 != 0}
print(output_dict)







# 2. 

import string

alphabet_dict = {letter: idx + 1 for idx, letter in enumerate(string.ascii_lowercase)}
print(alphabet_dict)
