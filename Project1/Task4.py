#I first imported everything I needed
#I then defined the addFolders function, this adds the folders using mkdirs(newPath)
#I have left in a default newPath to help with testing if you'd like
#This function will also delete the folder and recreate it with all the files if it already exists
#I then defined addFiles(), this is here if you wish to add files manually
#Then I defined renmaeFiles()
#It uses 2 for loops, the first one to change it all to lowercase, the second changes just the extension to uppercase
#Lastly I defined zipFile()
#This takes 2 paths and zips a folder of the first path into the second path




import os
from pathlib import Path
import shutil
import zipfile

 

def addFolders():
    answer = input("Do you wanna start a adding folders? y/n: ")

    keepGoing = (" ")
    if answer == ("y"):
            while keepGoing != ("quit"):

                    newPath = input("What is the path of the folder?: ")
                    #newPath = "/home/cadmin/AssignmentFolder
                    if os.path.isdir(newPath):
                        
                            shutil.rmtree(newPath)
                              
                            try:
                                os.makedirs(newPath)
                                
                                workingFolder = (newPath+"Working/")
                                backupFolder = (newPath+"Backup/")
                                
                                docsFolder = (workingFolder+"Docs/")
                                picsFolder = (workingFolder+"Pics/")
                                movieFolder = (workingFolder+"Movies/")

                                schoolFolder = (docsFolder + "School/")
                                partyFolder = (docsFolder + "Party/")

                                os.makedirs(workingFolder)
                                os.makedirs(backupFolder)
                                os.makedirs(docsFolder)
                                os.makedirs(picsFolder)
                                os.makedirs(movieFolder)
                                os.makedirs(schoolFolder)
                                os.makedirs(partyFolder)

                                coronaFile = (docsFolder+"CORONAVIRUS.txt")
                                open(coronaFile, "a").close()

                                dangerFile = (docsFolder+"DANGEROUS.txt")
                                open(dangerFile, "a").close()
        
                                safeFile = (docsFolder+"KEEPSAFE.txt")
                                open(safeFile, "a").close()

                                homeFile = (docsFolder+"STAYHOME.txt")
                                open(homeFile, "a").close()

                                hygeineFile = (docsFolder+"HYGEINE.txt")
                                open(hygeineFile, "a").close()

   
                                
                            except OSError:
                                print("Creation of directory %s failed\n" % newPath)
                            else:
                                print("Creation of directory %s successful\n" % newPath)
                    
                

                    else:
                        try:
                            os.makedirs(newPath)
                                
                            workingFolder = (newPath+"Working/")
                            backupFolder = (newPath+"Backup/")
                                
                            docsFolder = (workingFolder+"Docs/")
                            picsFolder = (workingFolder+"Pics/")
                            movieFolder = (workingFolder+"Movies/")

                            schoolFolder = (docsFolder + "School/")
                            partyFolder = (docsFolder + "Party/")

                            os.makedirs(workingFolder)
                            os.makedirs(backupFolder)
                            os.makedirs(docsFolder)
                            os.makedirs(picsFolder)
                            os.makedirs(movieFolder)
                            os.makedirs(schoolFolder)
                            os.makedirs(partyFolder)

                            coronaFile = (docsFolder+"CORONAVIRUS.txt")
                            open(coronaFile, "a").close()

                            dangerFile = (docsFolder+"DANGEROUS.txt")
                            open(dangerFile, "a").close()

                            safeFile = (docsFolder+"KEEPSAFE.txt")
                            open(safeFile, "a").close()

                            homeFile = (docsFolder+"STAYHOME.txt")
                            open(homeFile, "a").close()

                            hygeineFile = (docsFolder+"HYGEINE.txt")
                            open(hygeineFile, "a").close()

                            
                        except OSError:
                            print("Creation of directory %s failed\n" % newPath)
                        else:                            
                            print("Creation of directory %s successful\n" % newPath)
                    
                
                    keepGoing = input("Do you wanna add another folder? Type quit to stop: ")
                        

    else:
        print("You've chosen to not add folders\n")


def addFiles():
    answer = input("Do you wanna start a adding files to a folder? y/n: ")

    keepGoing = (" ")
    if answer == ("y"):
            while keepGoing != ("quit"):

                    filename = input("What is the name of the file?: ")
                    filePath = input("What is the path for the file?: ")
                    #filePath = "/home/cadmin/AssignmentFolder/Working/Docs/
                    
                    fullPath = os.path.join(filePath, filename + ".txt")
                    file1 = open(fullPath, "w")
                    toFile = input("What would you like to write into the file?: ")
                    
                    file1.write(toFile)
                    file1.close()
                
                    keepGoing = input("\nDo you wanna add another file? Type quit to stop: \n")

    else:
        print("You've chosen to not add files\n")


def renameFiles():
    answer = input("Do you wanna rename your files? y/n: ")

    if answer == ("y"):
        filePath = input("What is the path for the file?: ")
        #filePath = "/home/cadmin/AssignmentFolder/Working/Docs/
        if os.path.isdir(filePath):
        
            for file in os.listdir(filePath):
                os.rename(filePath + file, filePath + file.lower())
            
            for fname in os.listdir(filePath):
                name, ext = os.path.splitext(fname)
                os.rename(os.path.join(filePath, fname), os.path.join(filePath, name + ext.upper()))
                print("Your file have been renamed!")

        else:
            print("That path doesn't exist")


def zipFile():
    mypath = input("What is the path for the file?: ")
    foldername = "Docs"
    anotherpath = input("What is the path for the file?: ")
    os.chdir(anotherpath)
    zipf = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    zipFile(mypath, zipf)
    zipf.close()

addFolders()
addFiles()
renameFiles()
zipFile()
