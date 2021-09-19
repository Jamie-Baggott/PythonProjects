#I first get a number to begin with
#I think define the function reducer
#I set a simple if statment for whether it is odd or even
#Then I set a while loop to determine what to do going forward with the number until it reaches 0
#Finally I call the funciton at the bottom



n = int(input("Give a number: "))

def reducer(n):
   
   if (n % 2) == 0:
      print("{0} is Even".format(n))
   else:
      print("{0} is Odd".format(n))
   while n > 1:
       if (n % 2) == 0:
           n = (n//2)
           print(n)
       else:
               n = (n*3 + 1)
               print(n)
   
reducer(n)
