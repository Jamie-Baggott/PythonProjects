#Given a list of numberitems in a Python script, write a function,
#whichreceives the list as a parameter and sums all the numbersin the list,
#return andoutputs the sum.
#Sample list: [2, 4, 6, 7, 5]Expected Output: 24

#Implements a function that sums list of integers and return and output sum

#Define a function to sum list of intergers
def sumList(listVal):
    result = 0
    #for a in range(len(listVal)):
    for a in listVal:
        #result += listVal[a]
        #result = result + a
        result += a

    return result

#Define the input list
numList = [3, 4, 2, 6]

#Call function and pass parameter and get return value
output = sumList(numList)

#Output the result
print("Sum = ", output)
