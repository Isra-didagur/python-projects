def display_menu():
    print("\n checklist menu:")
    print("1. add items")
    print("2. view checklist")
    print("3. mark its as completed")
    print("4. remove item")
    print("5. exit")

def add_item(checklist):
    item = input("enter the item to add: ")
    checklist.append({"task":item,"completed":False})
    print({'item' "added to checklist,"})

def view_checklist(checklist):
    if not checklist:
        print("checklist is empty.")
        return
    print("\n your checklist:")
    for index,item in enumerate(checklist):
        status="[x]" if item ["completed"]else "[ ]"
        print(f"{index+1}.{status}{item['task']}")

def mark_completed(checklist):
    view_checklist(checklist)
    if not checklist:
        return
    try:
        choice=int(input("enter the number of items to mark as completed."))-1
        if 0<=choice<len(checklist):
            checklist[choice]["completed"]=True
            print(f"'{checklist[choice]['task']}'marked as completed")
        else:
            print("invalid choice.please enter a valid number")
    except ValueError:
        print("invalid input, please neter a number")

def remove_item(checklist):
    view_checklist(checklist)
    if not checklist:
        return
    try:
        choice=int(input("enter the number of item to remove: "))-1
        if 0<=choice<len(checklist):
            removed_item=checklist.pop(choice)["task"]
            print(f"'{removed_item}'remove from checklist")
        else:
            print("invalid choice.please enter a valid number")
    except ValueError:
        print("invalid input.please enter a number.")

def main():
    checklist=[]
    while True:
        display_menu()
        choice=input("enter your choice: ")

        if choice== '1':
            add_item(checklist)
        elif choice=='2':
            view_checklist(checklist)
        elif choice=='3':
            mark_completed(checklist)
        elif choice=='4':
            remove_item(checklist)
        elif choice=='5':
            print("exiting from the checklist application.")
            break
        else:
            print("invalid choice.please try again")

if __name__=="__main__":
    main()            
