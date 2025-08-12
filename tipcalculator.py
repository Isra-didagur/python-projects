print("welcome to the tip calculator.")
total_bill = float(input("what was the total bill? $"))
tip = int(input("what percentage tip would you like to give? 10, 12, or 15? ")) / 100
people = int(input("how many people to split the bill? ")) 
bill = total_bill * (1 + tip)
each = bill / people
print(f"each person should pay: ${round(each, 2)}")
