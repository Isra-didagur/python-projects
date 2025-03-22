def converttemp(temp,scale):
    if scale == "C":
            return (temp - 32) * 5/9
    elif scale == "F":
            return (temp * 9/5) +32
    else:
            return ("invalid scale")
    
def main():
        temp =float(input("enter the temperature:"))
        scale=input("enter the scale in capitals(C to F or F to C):")
        print(converttemp(temp,scale))
    
main()