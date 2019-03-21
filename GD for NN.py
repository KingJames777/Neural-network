import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_digits

def sigmoid(x):
      return 1/(1+np.exp(-x))

def dsigmoid(x):
      return x*(1-x)

class NeuralNetwork:
      def __init__(self,input_size,hidden_size,output_size):
            self.W1 = 0.01 * np.random.randn(input_size,hidden_size)#D*H
            self.b1 = np.zeros(hidden_size) #H
            self.W2 = 0.01 * np.random.randn(hidden_size,output_size)#H*C
            self.b2 = np.zeros(output_size)#C
        
      def loss(self,X,y,reg = 0.01):
            num_train, num_feature = X.shape
        #forward
            a1 = X  #input layer:N*D
            a2 = sigmoid(a1.dot(self.W1) + self.b1) #hidden layer:N*H
            a3 = sigmoid(a2.dot(self.W2) + self.b2) #output layer:N*C
        ##     y是向量化的
            loss = - np.sum(y*np.log(a3) + (1-y)*np.log((1-a3)))/num_train
            loss += 0.5 * reg * (np.sum(self.W1*self.W1)+np.sum(self.W2*self.W2)) / num_train
        
        #backward
            error3 = a3 - y #N*C
            dW2 = a2.T.dot(error3) + reg * self.W2  #(H*N)*(N*C)=H*C
            db2 = np.sum(error3,axis=0)
        
            error2 = error3.dot(self.W2.T)*dsigmoid(a2)    #N*H
            dW1 = a1.T.dot(error2) + reg * self.W1     #(D*N)*(N*H) =D*H
            db1 = np.sum(error2,axis=0)
        
            dW1 /= num_train
            dW2 /= num_train
            db1 /= num_train
            db2 /= num_train
        
            return loss,dW1,dW2,db1,db2
      
      def train(self,X,y,learn_rate=0.01,num_iters = 10000):
            ##此处y是向量化的，其余均未变动
            batch_size = 150
            num_train = X.shape[0]
        
            for i in range(num_iters):
                  batch_index = np.random.choice(num_train,batch_size,replace=True)
                  X_batch = X[batch_index]
                  y_batch = y[batch_index]
            
                  loss,dW1,dW2,db1,db2 = self.loss(X_batch,y_batch)
      
                  #update the weight
                  self.W1 += -learn_rate*dW1
                  self.W2 += -learn_rate*dW2
                  self.b1 += -learn_rate*db1
                  self.b2 += -learn_rate*db2
            
                  if i%500 == 0:
                        print(i,'\t',loss)
    
      def predict(self,X_test):
            a2 = sigmoid(X_test.dot(self.W1) + self.b1)
            a3 = sigmoid(a2.dot(self.W2) + self.b2)
            y_pred = np.argmax(a3,axis=1)
            return y_pred

digits = load_digits()
X = digits.data
y = digits.target
X -= np.mean(X,axis=0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)
y_train= LabelBinarizer().fit_transform(y_train) ##向量化

classify = NeuralNetwork(X.shape[1],50,10)
classify.train(X_train,y_train)
y_pred = classify.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("the accuracy is ",accuracy)

















