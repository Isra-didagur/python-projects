def mycalculator():
    print("\nWelcome to the calculator application.\n")
    print("Please enter the operation you want to perform:")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Exit")

def add(a,b):
    a=float(input("enter the 1st number: "))
    b=float(input("enter the 2nd number: "))
    result=a+b
    print(f"the sum of {a} and {b} is {result}")

def subtraction():
     num1=float(input("enter the 1st number: "))
     num2=float(input("enter the 2nd number: "))
     result=num1-num2
     print(f"the difference between  {num1} and {num2} is {result}")

def multiplication():
     num1=float(input("enter the 1st number: "))
     num2=float(input("enter the 2nd number: "))
     result=num1*num2
     print(f"the product of {num1} and {num2} is {result}")

def division():
      num1=float(input("enter the 1st number: "))
      num2=float(input("enter the 2nd number: "))
      result=num1/num2
      print(f"the answer for {num1} and {num2} is {result}")

def main():
     while True:
          mycalculator()
          choice=input("enter your choice: ")
          if choice=='1':
               add()
          elif choice=='2':
               subtraction()
          elif choice =='3':
               multiplication()
          elif choice=='4':
               division()
          elif choice=='5':
               print("\nexiting from the calculator operation.")
               break
          else:
               print("Invalid choice. Please enter a valid choice number.")
main()



        
            