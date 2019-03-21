import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_digits

def sigmoid(x):
      return 1/(1+np.exp(-x))


##为什么速度远低于隔壁的？还有，正则化应该有助于改进准确度。
class NNGD:
      def __init__(self,n_input,n_hidden,n_output):
            self.ih=np.random.randn(n_input,n_hidden) ## (n,hid)
            self.h=np.random.randn(n_hidden)  ## hid
            self.ho=np.random.randn(n_hidden,n_output)  ## (hid,k)
            self.o=np.random.randn(n_output)  ## k
            self.alpha=0.2

      def update(self,X,y):  ## X--n   y--k
            hid_output=sigmoid(X.dot(self.ih)-self.h)  ## hid
            y_output=sigmoid(hid_output.dot(self.ho)-self.o)  ## k
            g=y_output*(1-y_output)*(y-y_output)  ## k
            e=hid_output*(1-hid_output)*self.ho.dot(g)  ## hid
            self.ho+=self.alpha*hid_output[:,np.newaxis].dot(g[:,np.newaxis].T)
            self.o-=self.alpha*g
            self.ih+=self.alpha*X[:,np.newaxis].dot(e[:,np.newaxis].T)
            self.h-=self.alpha*e

      def train(self,X,y):
            n_iter=3000
            m,n=X.shape
            for j in range(n_iter):
                  for i in range(m):
                        self.update(X[i],y[i])

      def predict(self,X_test):
            hid_output=sigmoid(X_test.dot(self.ih)-self.h)  ## m,hid
            y_output=sigmoid(hid_output.dot(self.ho)-self.o)  ## m,k
            y_pred=np.argmax(y_output,axis=1)
            return y_pred,y_output

digits = load_digits()
X = digits.data
y = digits.target
X-= np.mean(X,axis=0)
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2,random_state=19901120,stratify=y)
y_train= LabelBinarizer().fit_transform(y_train) ##向量化

nngd=NNGD(X.shape[1],50,y_train.shape[1])
nngd.train(X_train,y_train)

y_pred,y_output=nngd.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("The accuracy is ",accuracy)  ##0.9833333333333333












