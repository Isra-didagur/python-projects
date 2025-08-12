import random as r

def hangman():
    print("Welcome to the hangman game")
    print("You've just been transported into a magical kingdom where words hold the power to save or doom someone. An evil wizard has captured a poor soul; the only way to free them is by guessing the secret word. Each wrong guess brings the prisoner closer to their fate.")
    
    word = "free"
    word = r.choice(word).lower()
    wordlength = len(word)
    display = '*' * wordlength
    display = list(display)
    
    print("".join(display))
    
    guessed_letters = set()
    max_attempts = 6
    attempts = 0
    correct_count = 0
    
    while attempts < max_attempts and correct_count < wordlength:
        guess = input("Enter letter: ").lower()
        
        if len(guess) != 1 or not guess.isalpha():
            print("Please enter a single letter.")
            continue
            
        if guess in guessed_letters:
            print("You already guessed that word.")
            continue
            
        guessed_letters.add(guess)
        
        found = False
        for i in range(wordlength):
            if word[i] == guess and display[i] == '*':
                display[i] = guess
                correct_count += 1
                found = True
        
        if found:
            print("Good guess!")
        else:
            attempts += 1
            print(f"Wrong guess! {max_attempts - attempts} attempts remaining.")
        
        print("".join(display))
        
    if correct_count == wordlength:
        print("Congratulations! You've saved the prisoner!")
    else:
        print(f"Game over. The word was {word}. The prisoner meets their fate.")

hangman()
