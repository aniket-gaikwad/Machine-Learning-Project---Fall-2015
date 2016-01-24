import csv
import collections
import itertools
import sys

csv.field_size_limit(sys.maxsize)

class csvHandler:
    """
    This class has methods that handles the CSV file operations
    """
    def __init__(self,dirName):
        self.dirName=dirName
        #self.fileName=None
        self.listOfbusiness=None
        self.data=None

    def readFile(self,fileName,flag):
        fr=csv.reader(open(fileName,"r"))
        firstline=True
        rowCnt=0
        bad_chars='[]"'''

        for row in fr:
            if firstline:
                firstline=False
                continue
            else:
                rowCnt+=1


    def writeFile(self,fileName,flag):
        print('\n Writting starts : ')
        #fileName=self.dirName+"/"+fileName
        fileName=fileName
        with open(fileName, 'wb') as fp:
            a = csv.writer(fp, delimiter=',')

            if flag==1:
                a.writerows(self.listOfbusiness)
            elif flag==2:
                a.writerows(self.data)
            else:
                print('\n ******** ERROR *************')
                print('\n Wrong Flag')


