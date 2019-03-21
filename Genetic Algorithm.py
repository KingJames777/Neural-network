import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
##from sklearn import preprocessing

##失败，还有这遗传算法运行也太慢了！

def sigmoid(x):
      return 1/(1+np.exp(-x))

class plot:
      def __init__(self,digits):
            self.digits=digits
      def digit(self):
            ##print(self.digits.images[200])  ##这是个二维矩阵
            ##a=np.zeros((8,8))
            ##a[:,4]=np.ones(8)*200
            ##a[3,:]=np.ones(8)*20
            ##print(a)

            ##plt.gray()
            plt.matshow(self.digits.images[200])
            plt.show()

class GA:
      def __init__(self,X_train,y_train):
            self.n_group=20  ##可行解集
            self.max_gen=200  ##最大代数
            self.mutation_prob=0.1  ##变异概率
            self.n_hidden=5
            self.X=X_train
            self.y=y_train
            self.m=X_train.shape[0] ##样本数
            self.n=X_train.shape[1]  ##属性数
            self.k=self.y.shape[1] ##输出向量长度，即类别数，即输出端点个数
            self.w_ih,self.w_ho,self.thre_h,self.thre_o=[],[],[],[]

      ##适应度
      def fitness(self,w_ih,thre_h,w_ho,thre_o):
            error=0
            for index in range(self.m):  ##逐个样本计算误差
                  hidden_input=self.X[index].dot(w_ih) ##可能会shape不匹配，已验证无碍
                  hidden_output=sigmoid(hidden_input-thre_h)
                  y_input=hidden_output.dot(w_ho)
                  y_output=sigmoid(y_input-thre_o)
##                  error+=(np.linalg.norm(y_output-self.y[index]))**2 ##norm是开过平方了
                  error+=-np.sum(self.y[index]*np.log(y_output)+(1-self.y[index])*np.log(1-y_output))/self.m
            return 1/(1+error)

      ##预测
      def predict(self,X_test):
            fit=[]
            pred=[]
            for i in range(self.n_group):
                  fit.append(self.fitness(self.w_ih[i],self.thre_h[i],self.w_ho[i],self.thre_o[i]))
            i=fit.index(max(fit))
            for index in range(len(X_test)):
                  hidden_input=X_test[index].dot(self.w_ih[i])
                  hidden_output=sigmoid(hidden_input-self.thre_h[i])
                  y_input=hidden_output.dot(self.w_ho[i])
                  y_output=sigmoid(y_input-self.thre_o[i])
                  pred.append(np.argmax(y_output))
            return pred
      
      ##初始化阈值和权重
      def initialize(self):
            for i in range(self.n_group):
                  self.w_ih.append(np.random.randn(self.n,self.n_hidden))
                  self.w_ho.append(np.random.randn(self.n_hidden,self.k))
                  self.thre_h.append(np.random.randn(self.n_hidden))
                  self.thre_o.append(np.random.randn(self.k))

      ##杂交
      def cross_over(self):
            shuf=list(range(self.n_group))
            np.random.shuffle(shuf)
            cw_ih,cw_ho,cthre_h,cthre_o=self.w_ih,self.w_ho,self.thre_h,self.thre_o ##复制
            for i in range(int(self.n_group/2)):  ##两两杂交
                  i1=2*i
                  i2=i1+1
                  a=np.random.randint(4)
                  lamda=np.random.randn()
                  b,c,d=np.random.randint(self.n),np.random.randint(self.n_hidden),np.random.randint(self.k)
                  if a==0:  ##杂交w_ih
                        temp=cw_ih[shuf[i1]][b][c]
                        cw_ih[shuf[i1]][b][c]=temp*(1-lamda)+lamda*cw_ih[shuf[i2]][b][c]
                        cw_ih[shuf[i2]][b][c]=cw_ih[shuf[i2]][b][c]*(1-lamda)+lamda*temp
                  elif a==1:  ##杂交w_ho
                        temp=cw_ho[shuf[i1]][c][d]
                        cw_ho[shuf[i1]][c][d]=temp*(1-lamda)+lamda*cw_ho[shuf[i2]][c][d]
                        cw_ho[shuf[i2]][c][d]=cw_ho[shuf[i2]][c][d]*(1-lamda)+lamda*temp
                  elif a==2: ##杂交thre_h
                        temp=cthre_h[shuf[i1]][c]
                        cthre_h[shuf[i1]][c]=temp*(1-lamda)+lamda*cthre_h[shuf[i2]][c]
                        cthre_h[shuf[i2]][c]=cthre_h[shuf[i2]][c]*(1-lamda)+lamda*temp
                  else:
                        temp=cthre_o[shuf[i1]][d]
                        cthre_o[shuf[i1]][d]=temp*(1-lamda)+lamda*cthre_o[shuf[i2]][d]
                        cthre_o[shuf[i2]][d]=cthre_o[shuf[i2]][d]*(1-lamda)+lamda*temp
            return cw_ih,cw_ho,cthre_h,cthre_o ##杂交后的100个解

      ##变异
      def mutation(self,cw_ih,cw_ho,cthre_h,cthre_o):
            shuf=random.sample(range(self.n_group),int(self.n_group*self.mutation_prob))
            for num in shuf:
                  a=np.random.randint(4)
                  lamda=np.random.randn()
                  b,c,d=np.random.randint(self.n),np.random.randint(self.n_hidden),np.random.randint(self.k)
                  if a==0:
                        cw_ih[num][b][c]=lamda
                  elif a==1:
                        cw_ho[num][c][d]=lamda
                  elif a==2: 
                        cthre_h[num][c]=lamda
                  else:
                        cthre_o[num][d]=lamda
            return cw_ih,cw_ho,cthre_h,cthre_o ##杂交再变异后的100个解

      ##选择
      def select(self,cw_ih,cw_ho,cthre_h,cthre_o):
            s_w_ih=cw_ih+self.w_ih
            s_w_ho=cw_ho+self.w_ho
            s_thre_h=cthre_h+self.thre_h
            s_thre_o=cthre_o+self.thre_o
            fit=[] ##适应值
            shuf=list(range(2*self.n_group))
            np.random.shuffle(shuf)
            for index in range(2*self.n_group):
                  fit.append(self.fitness(s_w_ih[index],s_thre_h[index],s_w_ho[index],s_thre_o[index]))
            for i in range(self.n_group):  ##  tournament
                  i1=shuf[2*i]
                  i2=shuf[2*i+1]
                  if fit[i1]>fit[i2]:
                        self.w_ih[i],self.w_ho[i]=s_w_ih[i1],s_w_ho[i1]
                        self.thre_h[i],self.thre_o[i]=s_thre_h[i1],s_thre_o[i1]
                  else:
                        self.w_ih[i],self.w_ho[i]=s_w_ih[i2],s_w_ho[i2]
                        self.thre_h[i],self.thre_o[i]=s_thre_h[i2],s_thre_o[i2]

if __name__ == '__main__':
      digits=datasets.load_iris()
      X=digits.data
      y=digits.target
      X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19901120,stratify=y)
      
      ##将输出转化成向量形式
      y_train=LabelBinarizer().fit_transform(y_train)

      ga=GA(X_train,y_train)
      ga.initialize()
      for i in range(ga.max_gen):
            cw_ih,cw_ho,cthre_h,cthre_o=ga.cross_over()
            acw_ih,acw_ho,acthre_h,acthre_o=ga.mutation(cw_ih,cw_ho,cthre_h,cthre_o)
            ga.select(acw_ih,acw_ho,acthre_h,acthre_o)

      pred=ga.predict(X_test)
      print(pred)
      print(accuracy_score(pred,y_test))
      



























