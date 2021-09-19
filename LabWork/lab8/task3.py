#Write a Python script that implements a function, which accepts a single list as parameter.
#The function should delete twoitemsfrom the list  and return the modified list.
#Interactively create the input list. After running the function, first output the input list and then the modified list contents.

#Implement a function to manipulate list items

import copy

#Define a function that accept list parameter and delete items
def itemDel(listVal):
    #make a copy of list since it is passed by reference
    newList = copy.copy(listVal)

    #Delete the first two items
    if len(newList) >= 2:
        for b in range(2):
            del newList[b]
    else:
        print("Input list must contain at least two items")

    return newList

#Interactively create input list
inputList = []
for c in range(6):
    value = input("Enter list item " + str(c+1) + ": ")
    inputList = inputList + [value]
    #inputList += [value]
    #inputList.append[value]

#Call the function to pass parameter and get the result]
result = itemDel(inputList)

#Output Contents
print("")
print("Input List: ", inputList)
print("Result List: ", result)
      
