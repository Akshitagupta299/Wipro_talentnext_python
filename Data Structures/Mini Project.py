#Find the runner up score

def find_runner_up(scores):
    unique_scores = list(set(scores))  
    unique_scores.sort(reverse=True)   
    return unique_scores[1]            
scores = [2, 3, 6, 6, 5]
runner_up = find_runner_up(scores)
print("Runner-up score:", runner_up)








#People and Fact

people_facts = {
    "Jeff": "Is afraid of Dogs.",
    "David": "Plays the piano.",
    "Jason": "Can fly an airplane."
}
print("Original facts:")
for person, fact in people_facts.items():
    print(f"{person}: {fact}")
print("\nUpdating facts...\n")
people_facts["Jeff"] = "Is afraid of heights."
people_facts["Jill"] = "Can hula dance."
print("Updated facts:")
for person, fact in people_facts.items():
    print(f"{person}: {fact}")








#Find the percentage

student_marks = {
    "Krishna": [67, 68, 69],
    "Arjun": [70, 98, 63],
    "Malika": [52, 56, 60]
}
student_name = input("Enter a name: ")
if student_name in student_marks:
    marks = student_marks[student_name]
    average = sum(marks) / len(marks)
    print(f"Average percentage mark: {int(average)}")
else:
    print("Student not found.")









#Find the name

input_string = input("Enter the sentence: ")
count = input_string.count("Alex")
print(count)

