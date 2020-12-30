#!/usr/bin/python3

nFiles = int(input("Enter the number of files: \n"))
fileType = str(input("Enter the file type (eg: main): \n"))
fileExt = str(input("Enter file extension: \n"))

for i in range(nFiles + 1):
    open(str(i) + "-" + fileType + fileExt, "w+")

print("Files creted successfully")
