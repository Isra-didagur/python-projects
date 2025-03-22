student={}

def grade(marks):
    if marks>=90:
        return "A"
    elif marks>=80:
        return"B"
    elif marks>=70:
        return"C"
    elif marks>=60:
        return"D"
    else:
        return"F"
    
def student():
    name=input("enter the name of the student: ")
    subject=int(input(f"enter the no.of subjects of the student {name} :"))
    marks=[]
    for i in range (subject):
        marks.append(int(input(f"enter the marks of the subject out of 100 subjects by commas {i+1} : ")))
    total=sum(marks)
    average=total/subject
    print(f"total marks obtained by {name} is {total} out of {subject*100}")
    print (f"average marks obtained by {name} is {average}")
    print(f"grade obtained by {name} is {grade(average)}")

student()
