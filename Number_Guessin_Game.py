import random as r
def numberguessinggame():
    print("welcome to the number guessing game")
    print("i am thinking of a number from 1 to 10")
    print("you have to uess the number in 3 tries")
    number = r.randint(1, 10)
    for i in range(3):
        guess =int(input("Enter the number: "))
        if guess == number:
            print("You have guessed it right")
            break
        else:
            print("Try again")
            print("The number was", number)
            print("Better luck next time")
numberguessinggame()
