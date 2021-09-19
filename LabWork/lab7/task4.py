#Write a Python script that will execute for a limited number of times specified by the user in a loop.
#In each iteration, use a function to generate and output six random numbers.
#The function should accept two parameters representing the range for the random number generation.
#Interactively request the user to provide the number of time to execute the script and the two numbers representing the range for the random generator.
#Remember to import the random module.


#Generate and output random numbers

import random

#Define function that accept two parameters and generate random numbers

def randomGen(lowerInterval, upperInterval):
    for a in range(6):
        #print("Interaction number is " + str(a))
        print(random.randint(lowerInterval, upperInterval))
        #return random.randint(lowerInterval, upperInterval)

#Request for inputs on how many times it will run and the ranges
numOfLoop = int(input("Provide the number of iterations: "))

#Request ranges for random number generator
lowerRange = int(input("Provide the lower interval range: "))
upperRange = int(input("Provide the upper interval range: "))

#Run the full logic
for b in range(numOfLoop):
    #Call the random generator function
    randomGen(lowerRange, upperRange)
    #print(randomGen(lowerRange, upperRange))
    print("Finished " + str(b) + " iteration")

print("Program execution completed! Thank you!")
