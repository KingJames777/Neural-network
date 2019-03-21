import numpy as np
from sklearn.preprocessing import LabelBinarizer


def rbf(x,c,beta):
      y=x-c
      return np.exp(-beta*y.dot(y))

x=np.random.randint(1,10,(5,))
c=np.random.randint(1,10,(5,))
print(x)


print(c)

beta=0.5

print(rbf(x,c,beta))
