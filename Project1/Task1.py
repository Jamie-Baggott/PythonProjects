#I have left in the options for you to automatically populate the list
#so that you can save time when testing or if by "interactively" make the list you
#didn't mean that you wanted them manually put in

#I used a while loop to keep adding to the list until it is told to stop.
#I then used a for loop to find the "?" character in each of the words that have been entered
#I think converted the list to a string to enable counting the common characters
#Using the counter it allows me to count the characters in the words





#Words = ["farshad","ghassemi?d","madam","?radar?","duration","con?tained"]

#n = int(input("How many words: "))

#for i in range(0,n):
#        word = input("Enter a word: ")
#        Words.append(word)


Words =[]

answer = input("Do you wanna start a list? y/n: ")

keepGoing = (" ")
if answer == ("y"):
        while keepGoing != ("stop"):

                word = input("Enter a name fam: ")
                keepGoing = input("Type stop to stop adding words: ")
        
                Words.append(word)
else:
        print("You haven't started a list\n")
        

print ("This is the list of words: ",Words, "\n")



Question = "?"
print("\nQuestion mark check!\n")

for i in Words:
    if Question in i : print(i + " contains the " + str(Question))

    

def listToString(s):
        str1 = " "

        return (str1.join(s))

s = Words
stringOfList =(listToString(s))

print(" ")



from collections import Counter

stringOfList = ' '.join(Words)
s1 = stringOfList.lower()

print("Common character check!\n")
commonLetters = Counter(s1)
for letters in commonLetters:
        print(f"The words contain {commonLetters[letters]} of the common character {letters}")


