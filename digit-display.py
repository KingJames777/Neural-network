from numpy import *
from os import listdir
import matplotlib.pyplot as plt

def img2vector(filename):
      res=[]
      file=open(filename)
      for i in range(32):
            line=file.readline()
            for j in range(32):
                  res.append(int(line[j]))
      return array(res)

def show_digit(X):
      m=len(X)
      plt.figure(figsize=(10,15))
      for i in range(m):
            plt.subplot(10,10,i+1)
            plt.axis('off')
            plt.imshow(X[i],cmap='binary')
      plt.savefig('digits.png')
      plt.show()

if __name__=='__main__':
      filelist=listdir('trainingDigits')
      X=[]
      k=0
      count=0
      for filename in filelist:
            if int(filename[0])==k:
                  count+=1
            else:
                  continue
            file=open('trainingDigits\%s'%filename)
            temp=[]
            for i in range(32):
                  line=file.readline()
                  nums=[]
                  for j in range(32):
                        nums.append(int(line[j]))
                  temp.append(nums)
            X.append(array(temp))
            if count==10:
                  k+=1
                  count=0
      show_digit(X)




















