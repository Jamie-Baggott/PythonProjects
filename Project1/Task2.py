#I started the lists for the words and new words
#I then used a for loop to work n amount of times so that the list is filled
#I set the number to be deleted and it randomly deletes that number of items in the list
#I then converted the new list into a tuple
#Finally I just printed both the original and new words


import random

Words = []
newWords = []

n = int(input("How many words: "))

for i in range(0,n):
        word = input("Enter a word: ")

        Words.append(word)

print (Words)  


DeleteNum = int(input("Give a number of items to delete: "))

remainingWords = len(Words) - DeleteNum
newWords = set(random.sample(Words, remainingWords))
newWords = [i for i in Words if i in newWords]


def convert(newWords):
    return tuple(newWords)

print("Result Tuple: " + str(convert(newWords)))
print("Original List: " + str(Words)
