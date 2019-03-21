import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_iris


##失败
def rbf(x,c,beta):
      y=x-c
      return np.exp(-beta*y.dot(y))

class RBFNN:
      def __init__(self,X,y,n_multiply):
            self.X=X
            self.y=y
            self.m,self.n=X.shape  ##  n--属性数，也即输入层个数
            self.k=y.shape[1]  ##  k--类别数，也即输出层个数
            self.n_multiply=n_multiply  ##每个类别选取的中心向量数
            self.init_center()
            self.w=np.random.randn(self.k,self.k*n_multiply)  ## 权重  k*h
            self.beta=np.random.randint(1,5,(self.k*n_multiply,))/1.0  ##beta--h

      ## 构造均值向量
      def init_center(self):
            res=[]
            for i in range(self.k):
                  index=np.where(self.y[:,i]==1)
                  temp=self.X[index]
                  size=temp.shape[0]
                  step=int(size/self.n_multiply)+1
                  for j in range(0,size,step):
                        res.append(np.mean(temp[j:j+step],axis=0))
            self.c=res

      def update(self,X,y,learning_rate=0.1):
            m,n=X.shape
            h=len(self.c)
            distance=np.empty((m,h))
            for i in range(m):
                  for j in range(h):
                        distance[i][j]=rbf(X[i],self.c[j],self.beta[j])
            output=distance.dot(self.w.T)  ##  m*k
            gra_w=(self.w.dot(distance.T)-y.T).dot(distance)  ## 权重矩阵的梯度

            ##计算beta的梯度
            gra_beta=[]
            for t in range(h):
                  ssum=0
                  for i in range(m):
                        for j in range(self.k):
                              y_ij=0
                              exp=(X[i]-self.c[t]).dot(X[i]-self.c[t])
                              dist=rbf(X[i],self.c[t],self.beta[t])
                              for s in range(h):
                                    exp1=(X[i]-self.c[s]).dot(X[i]-self.c[s])
                                    dist1=rbf(X[i],self.c[s],self.beta[s])
                                    y_ij+=self.w[j][s]*dist1
                              ssum+=self.w[j][t]*exp*dist*(y[i][j]-y_ij)
                  gra_beta.append(ssum)
            gra_beta=np.array(gra_beta)

            self.w-=learning_rate*gra_w
            self.beta-=learning_rate*gra_beta               

      def train(self,n_iter=300):
            batch_size=20
            for i in range(n_iter):
                  batch_index=np.random.choice(self.m,batch_size,replace=True)
                  X_batch=self.X[batch_index]
                  y_batch=self.y[batch_index]

                  self.update(X_batch,y_batch)
                  
      def predict(self,X_test):
            m,n=X_test.shape
            h=len(self.c)
            distance=np.empty((m,h))
            for i in range(m):
                  for j in range(h):
                        distance[i][j]=rbf(X[i],self.c[j],self.beta[j])
            output=distance.dot(self.w.T)  ##  m*k
            print(output)
            y_pred=np.argmax(output,axis=1)
            return y_pred

iris=load_iris()
X=iris.data
y=iris.target
X-= np.mean(X,axis=0)
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2,random_state=19901120,stratify=y)
y_train= LabelBinarizer().fit_transform(y_train)

rbfnn=RBFNN(X_train,y_train,3)
rbfnn.train()

y_pred=rbfnn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("The accuracy is ",accuracy)




























