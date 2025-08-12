def add(n1,n2):
    return n1 + n2

def subtract(n1,n2):
    return n1-n2

def multiply(n1,n2):
    return n1*n2

def divide(n1,n2):
    return n1/n2

operation= {
    "+": add,
    "-": subtract,
    "*": multiply,
    "/": divide,
    }

n1=input("enter the number: ")
for symbol in operation:
    print(symbol)
operation_symbol=input("pick the operation: ")
n2=input("enter the number: ")
ans=operation[operation_symbol](n1,n2)
print(f"{n1} {operation_symbol} {n2} = {ans}")