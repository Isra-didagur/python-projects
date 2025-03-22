def todolist():
    print("to do list")
    print("1.add task")
    print("2.view task")
    print("3.exit")
    Task = []; 
    while True:
        choice = input("enter the choice: ")
        if choice =='1':
            Task.append(input("enter the task:"))
        elif choice =='2':
            print(Task)
        elif choice == '3':
            break
        else:
            print("invalid choice")
todolist()