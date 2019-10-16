#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation


# In[2]:


def plot(X1_pos,X1_neg,X2_pos,X2_neg,w):
    plt.figure()
    plt.plot(X1_pos,X2_pos,'bo')    #positive class
    plt.plot(X1_neg,X2_neg,'r+')   #negative class
    x = np.linspace(-100, 100, 20)
    plt.plot(x, (5*x + 10)/(8), linestyle='solid', label="Target Function")   #target function
    a = w[1]
    b = w[2]
    c = w[0]
    plt.plot(x, (a*x + c)/(-b), linestyle='dashed',label="Hypothesis Function")   #classifier function
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Linearly Seperable Data")
    plt.legend()
    plt.show()


# In[3]:



def animateplot(w,X1_pos,X1_neg,X2_pos,X2_neg):
    axes.plot(X1_pos,X2_pos,'bo')    #positive class
    axes.plot(X1_neg,X2_neg,'r+')   #negative class
    x = np.linspace(-100, 100, 20)
    axes.plot(x, (5*x + 10)/(8), linestyle='solid', label="Target Function")   #target function
    
    a = w[1]
    b = w[2]
    c = w[0]
    
#     axes.xlabel("X1")
#     axes.ylabel("X2")
#     axes.title("Linearly Seperable Data")
#     axes.legend()
    axes.clear()
    axes.plot(x, (a*x + c)/(-b), linestyle='dashed',label="Hypothesis Function")   #classifier function


# In[4]:


#generating linearly seperable dataset
#line => ax +by +c  = 0
a = 5
b = -8
c = 10
def ispositive(x1,x2):
    sign = a*x1 + b*x2 + c
    if(sign>=0):  #postive class i.e. line>=0
        return True
    return False
    


# In[5]:


def dataset_construction(size):
    pos_class_size = random.randrange(size/4,3*size/4,1)
    neg_class_size = size - pos_class_size
    print("positive class size",pos_class_size)
    print("negative class size",neg_class_size)
    pos_cnt = 0
    neg_cnt = 0
    datalist = []
    while(pos_cnt<pos_class_size or neg_cnt<neg_class_size):
        x1 = random.randrange(-100,100,1)
        x2 = random.randrange(-100,100,1)
        if(ispositive(x1,x2)):
            if(pos_cnt<pos_class_size):
                pos_cnt+=1
                datalist.append([x1,x2,1])
        else:
            if(neg_cnt<neg_class_size):
                neg_cnt+=1
                datalist.append([x1,x2,-1])  
    dataset = np.array(datalist).reshape(size,3)
    return dataset


# In[6]:


#segregating dataset acc. to class
dataset = dataset_construction(20)
X1_pos = []
X2_pos = []
X1_neg = []
X2_neg = []
for i in range(20):
    if(dataset[i][2]==1):
        X1_pos.append(dataset[i][0])
        X2_pos.append(dataset[i][1])
    else:
        X1_neg.append(dataset[i][0])
        X2_neg.append(dataset[i][1])


# In[7]:


plt.figure()
plt.plot(X1_pos,X2_pos,'bo')    #positive class
plt.plot(X1_neg,X2_neg,'r+')   #negative class
x = np.linspace(-100, 100, 20)
plt.plot(x, (a*x + c)/(-b), linestyle='solid',label = "Target Function")   #target function
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.title("Linearly Seperable Data")


# In[8]:


w0 = random.random()
w1 = random.random()
w2 = random.random()
w = np.array([w0,w1,w2])
#alternate ways
# w = np.random.rand(1,3)


# In[9]:


#adding x0 in X
x = np.ones((20,3))
x[:,1:] = dataset[:,:-1]
y = np.ones((20,1))
y = dataset[:,2]
w_transition = []
w_transition.append(w)


# In[20]:


#perceptron learning algorithm
def perceptron_learning_algo(X,Y,learning_rate,w):
    misclassified_samples  = True
    iterations = 0
    fig = plt.figure()
    x = np.linspace(-100, 100, 20)
    plt.ylim(-100,100)
    plt.plot(x, (5*x + 10)/(8), linestyle='solid', label="Target Function")   #target function
    a = w[1]
    b = w[2]
    c = w[0]
    plt.plot(x, (a*x + c)/(-b), linestyle='dashed')   #classifier function
    while(misclassified_samples):
        misclassified_samples = False
        iterations+=1
        for idx,x in enumerate(X):
            prod = w.dot(x)
            if(-1*prod*Y[idx] > 0):    #misclassification problem
                misclassified_samples = True
                w = w + learning_rate*x*Y[idx]
                w_transition.append(w)
                
                a = w[1]
                b = w[2]
                c = w[0]
                x_ = np.linspace(-100, 100, 20)
                plt.plot(x_, (a*x_ + c)/(-b), linestyle='dashed')   #classifier function
                plt.pause(0.05)
#                 axes.clear()
#                 animateplot(axes,X1_pos,X1_neg,X2_pos,X2_neg,w)
#     ani = animation.FuncAnimation(fig,animateplot,interval = 1000)
    a = w[1]
    b = w[2]
    c = w[0]
    x = np.linspace(-100, 100, 20)
    plt.plot(x, (a*x + c)/(-b), linestyle='solid',label="Hypothesis Function")   #classifier function
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()
    return (w,iterations,w_transition)                


# In[21]:


W,iterations,W_transition = perceptron_learning_algo(x,y,1,w)

def plotting(W):
    for w_ in W:
        a = w_[1]
        b = w_[2]
        c = w_[0]
        x = np.linspace(-100, 100, 20)
        axes.plot(x, (5*x + 10)/(8), linestyle='solid', label="Target Function")   #target function
    
        axes.plot(x, (a*x + c)/(-b), linestyle='dotted')   #classifier function
    
def animate(i):
    axes.plot(X1_pos,X2_pos,'bo')    #positive class
    axes.plot(X1_neg,X2_neg,'r+')   #negative class
    axes.clear()
    plotting(W_transition)
# ani = animation.FuncAnimation(fig,animate,interval = 10000)
# plt.show()


# In[ ]:





# In[39]:


plot(X1_pos,X1_neg,X2_pos,X2_neg,W)


# In[12]:


print("No. of iterations",iterations)


# In[13]:


###same with size of 100
dataset = dataset_construction(100)
# print(dataset)
X1_pos = []
X2_pos = []
X1_neg = []
X2_neg = []
for i in range(100):
    if(dataset[i][2]==1):
        X1_pos.append(dataset[i][0])
        X2_pos.append(dataset[i][1])
    else:
        X1_neg.append(dataset[i][0])
        X2_neg.append(dataset[i][1])


# In[14]:


plt.figure()
plt.plot(X1_pos,X2_pos,'bo')    #positive class
plt.plot(X1_neg,X2_neg,'r+')   #negative class
x = np.linspace(-100, 100, 20)
plt.plot(x, (a*x + c)/(-b), linestyle='solid',label = "Target Function")   #target function
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.title("Linearly Seperable Data")


# In[15]:


w0 = random.randrange(0,100,1)
w1 = random.randrange(0,100,1)
w2 = random.randrange(0,100,1)
w = np.array([w0,w1,w2])
#alternate ways
# w = np.random.rand(1,3)


# In[16]:


#adding x0 in X
x = np.ones((100,3))
x[:,1:] = dataset[:,:-1]
y = np.ones((100,1))
y = dataset[:,2]
w_transition = []
w_transition.append(w)


# In[17]:


W,iterations,W_transition = perceptron_learning_algo(x,y,1,w)


# In[18]:


plot(X1_pos,X1_neg,X2_pos,X2_neg,W)


# In[19]:


print("No. of iterations",iterations)


# In[ ]:




