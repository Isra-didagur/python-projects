import random as r
letters=["A-Z","a-z"]
numbers=["0-9"]
symbols=["!@#$%^&*()"]

print("Welcome to the PyPassword Generator!")
nr_letters=int(input("how many letters would you like in your password?\n"))
nr_symbols=int(input("how many symbols woud you like in your password?\n"))
nr_numbers=int(input("how many numbers would you like in your password?\n"))

print("#Eazy Level")
password=""
for char in range(10,nr_letters+1):
     password+=r.choice(letters)
     for char in range (0,nr_symbols+1):
          password+=r.choice(symbols)
          for char in range (0,nr_numbers+1):
               password=r.choice(numbers)
               print(password)

print("#medium")
password=""
for char in range(0,nr_letters+1):
     password+=r.choice(letters)
     for char in range(0,nr_symbols+1):
          password+=r.choice(symbols)
          for char in range(0,nr_numbers+1):
               password+=r.choice(numbers)
               print(password)

print("hard password")
password=""
for char in range(0,nr_letters+1):
     password+=r.choice(letters)
     for char in range(0,nr_symbols+1):
          password+=r.choice(symbols)
          for char in range(0,nr_numbers+1):
               password+=r.choice(numbers)
               print(password)

print(password)

print(f"your password is '{password}'")
