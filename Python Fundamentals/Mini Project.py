#Ride on Miles (mini project 1)

def travel_advice():
    distance = float(input("How many miles do you want to travel? "))

    if distance < 3:
        print("I suggest Bicycle to your destination.")
    elif distance < 300:
        print("I suggest Motor-Cycle to your destination.")
    else:
        print("I suggest Super-Car to your destination.")

travel_advice()










#Ride on Miles (mini project 2)

def server_costs():
    hourly_rate = 0.51 
    hours_per_day = 24
    daily_cost = hourly_rate * hours_per_day
    weekly_cost = daily_cost * 7
    monthly_cost = daily_cost * 30 
    budget = 918
    days_with_budget = budget / daily_cost
    print(f"Cost to operate one server per day: ${daily_cost:.2f}")
    print(f"Cost to operate one server per week: ${weekly_cost:.2f}")
    print(f"Cost to operate one server per month: ${monthly_cost:.2f}")
    print(f"With $918, you can operate one server for {int(days_with_budget)} days.")

server_costs()