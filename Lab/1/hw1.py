import numpy as np
from scipy import linalg as li
from numpy.linalg import matrix_rank
import random as rand 
import math
def ex1():
    A = np.array([[1,2,3],[2,1,4]])
    B = np.array([[1,0],[2,1],[3,2]])
    C = np.array([[3,-1,3],[4,1,5],[2,1,3]])
    D = np.array([[2,-4,5],[0,1,4],[3,2,1]])
    E = np.array([[3,-2],[2,4]])
    print("a)")
    print(A)
    a = (2*A).transpose()
    print(a)
    print("b)")
    try:
        b = A - B
        print(b.transpose())    
    except Exception as t:
        print(t)
    print("c)")
    try:
        c = (3*(B.transpose()) - A)
        print(c.transpose())
    except Exception as e:
        print(e)
    print("d)")
    try:
        d = (-A).transpose()
        d1 = d.dot(E)
        print(d1)
    except Exception as e:
        print(e)
    print("e)")
    try:
        e = (C + 2*(D.transpose()) + E).transpose()
        print(e)
    except Exception as e:
        print(e)
def ex2():
    A = [[1,2],[3,2]]
    B = [[2,-1],[-3,4]]
    try:
        if(A.dot(B) == B.dot(A)):
            print("equal")
        else:
            print("unequal")
    except:
        print("dimensions not equal")

def ex3():
    v1 = np.array([-2,0,1])
    v2 = np.array([0,1,0])
    v3 = np.array([2,0,4])
    V = [v1,v2,v3]
    orthogonal_set = True
    ##orthogonal set if two distinct vectors' dot product is 0
    print("Checking euclidean inner product of every pair of vectors....")
    for i in range(3):
        vec1 = V[i]
        for j in range(3):
            if(i!=j):
                vec2 = V[j]
                vec3 = (vec1).dot(vec2)  ##since v1^T.v2 = v2^T.v1
                n = vec3.shape
                if(vec3!=0):   ##if any dot product is non-zero then it is not orthogonal set
                    orthogonal_set = False
    print("Is orthogonal set",orthogonal_set)
    print("Calculating magnitude of vectors...")
    V_mag = [np.sqrt((v1.transpose()).dot(v1)),np.sqrt((v2.transpose()).dot(v2)),np.sqrt((v3.transpose()).dot(v3))]
    unit_vectors = True
    orthonormal_set = False
    for i in range(3):
        print(V_mag[i])
        if(V_mag[i]!=1):
            unit_vectors = False
    if(orthogonal_set and unit_vectors):
        orthonormal_set = True
    print("Is orthonormal set",orthonormal_set)

    print("To turn the vectors set into orthonormal set the magnitude of vectors should be 0")
    v1 = (1/V_mag[0])*v1
    v2 = (1/V_mag[1])*v2
    v3 = (1/V_mag[2])*v3
    print( "New v1=",v1)
    print("New v2=",v2)
    print("New v3=",v3)
def ex4():
    m = int(input("Enter m"))
    n = int(input("Enter n"))
    listm = []
    listn = []
    rand.seed(0)
    for i in range(m):
        x  = rand.random()
        listm.append(x)
    rand.seed(3)
    for i in range(n):
        y  = rand.random()
        listn.append(y)
    X = np.array(listm).reshape(m,1)
    Y = np.array(listn).reshape(n,1)
    R = X.dot(Y.transpose())
    print("The rank of matrix is",matrix_rank(R))
    
    
def ex5():
    m = int(input("Enter m"))
    n = int(input("Enter n"))
    p = int(input("Enter p"))
    listx = []
    X = []
    rand.seed(0)
    for i in range(n):
        x = []
        for j in range(m):
            e = rand.random()
            x.append(e) 
        listx.append(x)
        x = np.array(x).reshape(m,1)
        X.append(x)
        #print(x)
    X = np.array(X).reshape(m,n)
    print("X = ",X)
    listy = []
    YT = []
    rand.seed(3)
    for i in range(n):
        y = []
        for j in range(p):
            e = rand.random()
            y.append(e)
        listy.append(y) 
        y = np.array(y).reshape(p,1)
        YT.append(y)
        #print(y)
    YT = np.array(YT).reshape(p,n)
    print("YT = ",YT)
    Y = YT.transpose()
    print("Y = ",Y)
    LHS = X.dot(Y)
    print("LHS = XY  =",LHS)
    sum = np.zeros((m,p))   
    for i in range(n):
        print("x(i)= ",X[:,i],"y(i)=",YT[:,i])
        sum += np.array(X[:,i]).reshape(m,1).dot(np.array(YT[:,i]).reshape(p,1).transpose())
    print("RHS =",sum)
    if(LHS.all()==sum.all()):
        print("Equal")
    else:
        print("Unequal")
def ex6():
    m = int(input("Enter m"))
    n = int(input("Enter n"))
    X = np.random.randint(0,10,(m,n))
    print(X)
    P = (X.transpose()).dot(X)
    print("Is Symmetric",np.allclose(P, P.transpose(), rtol=1e-05, atol=1e-08))
    print("Is positive-semidefinite?",np.all(np.linalg.eigvals(P) >= 0))
    print("Is positive-definite?",np.all(np.linalg.eigvals(P) > 0))
def ex7(x,y):
    
    f = math.exp(x) + math.exp(y) + math.exp(-2*x*y) - math.log(-1*x*x*y,math.exp(1))
    difffx = math.exp(x) + -2*y*math.exp(-2*x*y) + (1/(-1*x*x*y))*(-2*x*y)
    difffy = math.exp(y) + -2*x*math.exp(-2*x*y) + (1/(-1*x*x*y))*(-x*x)
    print("partial differentiation wrt x",difffx)
    print("partial differentiation wrt y",difffy)
    
def ex8():
    A = np.array([[2,1,3],[1,1,2],[3,2,5]],dtype=complex) #datatype = complex
    #eigen-decomposition
    eigval,eigvec = li.eig(A)
    print("eigen values ",eigval)
    print("eigen vector",eigvec)
    B = A.dot(eigvec[:,0])
    print(B)
    C = eigval[0]*eigvec[:,0]
    print(C)
    if(C.all()==B.all()):
        print("Eigen-Decomposition done")
    Q  = eigvec
    Qinv = li.inv(Q)
    L = np.diag(eigval)
    A2 = Q.dot(L).dot(Qinv)
    print("A=",A,"= QLQ^-1 = ",A2,"Is Equal",A.all()==A2.all())

    #rank of A
    rank = matrix_rank(A)
    print("rank = ",rank)
    #positive semidefinite/definite
    print("Is positive-semidefinite?",np.all(np.linalg.eigvals(A) >= 0))
    print("Is positive-definite?",np.all(np.linalg.eigvals(A) > 0))
    #singularity
    determinant = li.det(A)
    print(determinant)
    singular = determinant==0
    print("Is Singular",singular)
    
if __name__ == "__main__":
    print("\n1------------------------------\n")
    ex1()
    print("\n2------------------------------\n")
    ex2()
    print("\n3------------------------------\n")
    ex3()
    print("\n4------------------------------\n")
    ex4()
    print("\n5------------------------------\n")
    ex5()
    print("\n6------------------------------\n")
    ex6()
    print("\n7------------------------------\n")
    ex7(3,-5)
    print("\n8------------------------------\n")
    ex8()
    


