#!/usr/bin/env python
# coding: utf-8

# This file defines several functions used for the code of the paper 
# "One Arrow, Two Kills: A Unified Framework for Achieving Optimal Regret Guarantees in Sleeping Bandits".

import numpy as np
import random
from time import sleep
from tqdm import tqdm
from scipy import linalg as LA
from matplotlib import pyplot as plt 
from scipy.stats import mstats

random.seed(1)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ###  UCB Algorithm
# 
# n: vector of dim k stating number of pulls of each arm\
# z: vector of dim k stating cumulative loss of each arm\
# t: number of rounds\
# avail: vector of dim k stating availabilities\
# loss: instantaneous loss vector of the arms\
# \
# return: chosen arm, and updated value for n, z, and t
# 

# In[2]:


def ucb(n, z, t, avail, loss):
    if 0 in n[avail==1]: # some arm is available and has not been pulled yet
        iaux = random.randint(0,np.sum((n==0)*(avail==1))-1)
        choice = np.where((n==0)*(avail==1))[0][iaux] # select one of these arms at random
    else:
        lmin = np.min((z/np.maximum(n,1)-np.sqrt(8*np.log(t+1)/np.maximum(n,1)))[avail==1])
        aux = (z/np.maximum(n,1)-np.sqrt(8*np.log(t+1)/np.maximum(n,1)) == lmin)*(avail==1)
        iaux = random.randint(0,np.sum(aux)-1)
        choice = np.where(aux)[0][iaux]
        
    z[choice]+=loss[choice]
    n[choice]+=1
    
    return choice, n, z


# ### Internal regret algorithm
# 
# L: matrix of size K x K containing the cumulative losses of experts i->j
# 

# In[3]:


########## INTERNAL REGRET ALGORITHM       

def SIEXP3(L, eta, avail, loss):
    
    K = len(L)
    k = np.sum(avail)
    
    if k>1:
        idx = np.array(np.where(avail>0)).reshape(k,1)
        M = L[idx,idx.T]
        
        min_loss = 0
        if len(np.where(M>0)[0])>0:
            min_loss = np.min(M[np.where(M>0)])

        # define the exponential matrix Q
        Q = np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                if i!=j:
                    Q[i][j]=np.exp(-eta*(M[i][j] - min_loss))

        # normalize entries
        Qt=np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                Qt[i][j]=Q[i][j]/np.sum(Q[j])

        v, w = LA.eig(np.transpose(Qt))
        w=np.transpose(w)

        probs=np.absolute(np.real(w[np.argmin(np.absolute(v - 1))]))
    else:
        probs = 1
    
    p = np.zeros(K)
    p[np.where(avail>0)] = probs/np.sum(probs)
    
    choice=np.random.choice(a=K,size=1,p=p)

    expert_probs=np.zeros((K,K,K))
    for i in range(K):
        for j in range(K):
            if i!=j:
                temp=p.copy()
                temp[i]=0
                temp[j]=p[i]+p[j]
                expert_probs[i][j]=temp

    est_loss=np.zeros(K)
    est_loss[choice]=loss[choice]/p[choice]
    
    expert_loss=np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if i!=j:
                if avail[j] == 1:
                    expert_loss[i][j]=np.dot(expert_probs[i][j],est_loss)
                else:
                    expert_loss[i][j]=loss[choice]

    L+=expert_loss
    return choice, L, p 


# ### Sleeping Exp3 from ICML paper

# In[4]:


def projected_prob(p,s):
    q=np.multiply(p,s)/np.sum(p[s==1])
    return q


# In[5]:


def sleeping_exp3(cum_loss,S,eta,lam,avail,loss):
    
    K = len(cum_loss)
    p = np.exp(-eta * (cum_loss - np.min(cum_loss)))
    q = projected_prob(p,avail)
    choice = np.random.choice(a=K,size=1,p=q)
    S = np.append(S, avail.reshape((1,K)), axis=0)

    q_bar = np.zeros(K)
    for i in range(len(S)):
        q_bar = i/(i+1) * q_bar + 1/(i+1) * projected_prob(p, S[i])
    
    est_loss=np.zeros(K)
    est_loss[choice]=loss[choice]/(q_bar[choice]+lam)
    
    cum_loss+= est_loss
    return choice, cum_loss, S

