name=input("what is your name?\n")
print(f"Hello {name}")
price=int(input("what is your bid:? $"))

bids={}
bids[name]=price

should_continue=input("Are there any other bidders? Type 'yes' or 'no'.\n")
while should_continue=="yes":
  name=input("what is your name?\n")
  print(f"Hello {name}")
  price=int(input("what is your bid:? $"))
  bids[name]=price
  should_continue=input("Are there any other bidders? Type 'yes' or 'no'.\n")
  if should_continue=="no":
    break
print(bids)
highest_bid=0
for key in bids:
  if bids[key]>highest_bid:
    highest_bid=bids[key]
    winner=key
print(f"The winner is {winner} with a bid of ${highest_bid}")