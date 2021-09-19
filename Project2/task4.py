#Jamie Baggott R00149982

keepGoing = ("1")
while keepGoing != ("finish the work"):
    keepGoing = input("\nType the phrase finish the work to quit:  ")
    if keepGoing != ("finish the work"):
            inputSentence = input("Enter a sentence:  ")
            inputNum = int(input("Enter a number?: "))

            allCapital = inputSentence.title()
            firstCapital = inputSentence.capitalize()
            if len(inputSentence) >= 10:
                print("Every word will now have it's first letter capitalized")
                print(allCapital)

            if len(inputSentence) < 9:
                print("Only the first word will be capitalized")
                print(firstCapital)
    if keepGoing == ("finish the work"):
        print("Program completed, thank you!")
