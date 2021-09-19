#Given a tuple of at least 6 numbers(can be manually or interactively created)in a Python script,
#write a function that accepts the tuple and a second parameteras  arguments.
#The function should insert the value of the second parameterat the 4thindex of the tuple and append the first three items in the tuple to the end of the tuple.
#Print out the original and the changed tuples. Sample original tuple: (2, 4, 6, 7, 5, 4, 9, 8)
#Expected Output: (2, 4, 6, 7, var, 5, 4, 9, 8, 2, 4, 6) where varrepresents the second parametervalue.

#Use a function to manipulate tuple and insert value

#Define function that accept two parameters - a tuple and a value
def insertVal(tupleVal, val):
    #Covert to a mutable data type
    editTuple = list(tupleVal)

    #Insert specificed value into the list
    editTuple.insert(4, val)

    #Append the first 3 values
    #editTuple.append(editTuple[:3])
    for b in range(3):
        editTuple.append(editTuple[b])

    #Convert back to tuple and return value
    return tuple(editTuple)

#Interactively generate input list.
inputList = []
for a in range(7):
    value = input("Enter list item " + str(a+1) + ": ")
    inputList += [value]

#Convert to Tuple
tupleItems = tuple(inputList)

#Request for the value to be inserted. It must be integer value
intVal = 0
while True:
    try:
        inVal = int(input("Enter a number to be inserted: "))
        break
    except ValueError:
        print("Invalid input. Please enter a number")

#Call the function and pass parameters
result = insertVal(tupleItems, inVal)

#Output value
print("Original Tuple: ", tupleItems)
print("Edited Tuple: ", result)
