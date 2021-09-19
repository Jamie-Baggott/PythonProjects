#Write a Python script that prompts for and reads two integer values, and also prompt for and reads an operator.
#One of +, -, *, /. Carry out the arithmetic operation on the two numbers.
#Print out the formula used and the answer.
#A sample run should look:Enter a number: 15Enter another number: 2Enter one of +, -, *, /:/15 / 2 = 7.5Tip:
#Use an if statement to check the inputted operator.



#Demand for two integer numbers and an arithmetic operator to be exectuted

#Request for first input adn store in a variable
num1 = int(input("Enter a number: "))

#Request for the second number and store it
num2 = int(input("Enter another number: "))

#Request the operator option
operator = input("Enter one of +, *, - or / : ")

#Implement the logic for execution
if operator == "+":
    result = num1 + num2

    print("Output: " + str(num1) + " + " + str(num2) + " = " + str(result))
    print("Output: ", num1, " + ", num2, " = ", result)
    print("Output: {} + {} = {}".format(num1, num2, result))

elif operator == "*":
          result = num1 * num2
          print("Output: " + str(num1) + " * " + str(num2) + " = " + str(result))

elif operator == "-":
          result = num1 - num2
          print("Output: " + str(num1) + " - " + str(num2) + " = " + str(result))

elif operator == "/":
          result = num1 / num2
          print("Output: " + str(num1) + " / " + str(num2) + " = " + str(result))

else:
          print("Invalid operator entered")
            
