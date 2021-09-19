#Jamie Baggott R00149982


numberList = []
resultList = []
fullList = []
addName = ("1")
while addName != ("stop"):
    addName = input("\n1. Enter PPS Number \n2. Enter sick to find all sick patients \n3. Enter healthy to see all negative patients \n4. Enter stop to quit: ")
    if addName == ("1"):
            ppsNum = input("Enter a PPS number:  ")
            result = input("Are they negative or positive?: ")
            fullDetails = (str(ppsNum) + " is " + str(result)) 
            if ppsNum in numberList:
                    print(str(fullDetails))
            elif ppsNum not in numberList:
                        numberList = numberList + [str(ppsNum)]
                        resultList = resultList + [str(result)]
                        fullList = fullList + [str(fullDetails)]
                
                        print("\n", fullList, "\n")

    if addName == ("sick"):
        negativeCases = resultList.count("negative")
        print("\nThe amount of negative cases are ", str(negativeCases))
                
    if addName == ("healthy"):
        positiveCases = resultList.count("positive")
        print("\nThe amount of positive cases are ", str(positiveCases))
    
    if addName == ("stop"):
        print("Thank you for using the program")

