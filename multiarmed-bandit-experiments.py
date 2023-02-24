#!/usr/bin/env python
# coding: utf-8

## Sleeping Multi-armed bandit experiments


# This file reproduces the multi-armed bandit experiments of the paper "One Arrow, Two Kills: A Unified Framework for
# Achieving Optimal Regret Guarantees in Sleeping Bandits".


from algorithms import *


# ### Experiment 1
# This is a toy experiment which is not in the paper
# K = 10 actions
# Fixed losses : (1/K,2/K,...,1) 
# i.i.d. availabilities for each action



np.random.seed(1)
T = 1000
K = 10
niter = 20
sampling_rules = ["UCB","SI-EXP3","S-EXP3"]

regret_ordering = np.zeros((3,niter,T))
regret_external = np.zeros((3,niter,T))
regret_internal = np.zeros((3,niter,T))
choice = np.zeros(3,dtype=np.int8)

for i in tqdm(range(niter)):
    regret_external_aux = np.zeros((3,K))
    regret_internal_aux = np.zeros((3,K,K))
    
    sleep=np.random.uniform(low=0.3,high=0.9,size=K)
    n=np.zeros(K)
    z=np.zeros(K)
    L=np.zeros((K,K))

    S = np.empty((0,K), int)
    cum_loss =np.zeros(K)

    for t in range(T):
        loss=1-np.array(range(K),float)/K
        avail=np.array(np.random.binomial(n=1,p=sleep,size=K))
        while 1 not in avail:
            avail=np.array(np.random.binomial(n=1,p=sleep,size=K))

        eta = 1/(t+1)**0.5
        lam = 1/np.sqrt(t+1)

        choice[0], n, z = ucb(n, z, t, avail, loss)
        choice[1], L, p = SIEXP3(L, eta, avail, loss)
        choice[2], cum_loss, S = sleeping_exp3(cum_loss,S,eta,lam,avail,loss)
        oracle_choice = np.max(np.where(avail >0))

        for j in range(3):
            regret_ordering[j,i,t] = loss[choice[j]]-loss[oracle_choice]
            regret_external_aux[j,:] += (loss[choice[j]]-loss)*avail
            regret_internal_aux[j,choice[j],:] += (loss[choice[j]]-loss)*avail
            regret_external[j,i,t] = np.max(regret_external_aux[j,:])
            regret_internal[j,i,t] = np.max(regret_internal_aux[j,:,:])
    
    for j in range(3):
        regret_ordering[j,i,:] = np.cumsum(regret_ordering[j,i,:])


sampling_rules = ["UCB","SI-EXP3","S-EXP3"]
regret_names = ["Policy", "External", "Internal"]
regrets = np.zeros((3,3,niter,T))
regrets[0,:,:,:] = regret_ordering
regrets[1,:,:,:] = regret_external
regrets[2,:,:,:] = regret_internal

T_plot = T
for k in range(3):
    fig=plt.figure(figsize=(4,3), dpi= 100, facecolor='w', edgecolor='k')  # to have big plots
    for i in range(3):
        means = np.mean(regrets[k,i,:,:], axis=0)
        quantiles = mstats.mquantiles(regrets[k,i,:,:], axis=0)
        plt.plot(np.arange(T_plot-1),means[1:T_plot], label=sampling_rules[i], color = colors[i+1], linewidth=2)
        plt.fill_between(np.arange(T_plot-1), quantiles[0,1:T_plot] ,quantiles[2,1:T_plot], color= colors[i+1], alpha=.2)
    plt.xlabel("Time")
    plt.ylabel(regret_names[k]+' Regret')
    plt.legend()
    plt.savefig('exp1_'+regret_names[k]+'.pdf', bbox_inches = "tight")


# ### Experiment 2 
# Dependent losses and availabilities
# This experiment reproduces the Dependent environment experiment of Figure 2.


np.random.seed(2)
sampling_rules = ["UCB","SI-EXP3","S-EXP3"]

T = 1000
K = 3
niter = 20

regret_ordering = np.zeros((3,niter,T))
regret_external = np.zeros((3,niter,T))
regret_internal = np.zeros((3,niter,T))
choice = np.zeros(3,dtype=np.int8)

for i in tqdm(range(niter)):
    regret_external_aux = np.zeros((3,K))
    regret_internal_aux = np.zeros((3,K,K))
    mu=np.random.uniform(low=0,high=1,size=K)
    n=np.zeros(K)
    z=np.zeros(K)
    L=np.zeros((K,K))

    S = np.empty((0,K), int)
    cum_loss =np.zeros(K)



    for t in range(T):

        a=random.randint(0,K)            
        if a==0:
            avail=np.array([1,1,0])
            loss=np.array([0,.5,1],float)
        elif a==1:
            avail=np.array([1,1,1])
            loss=np.array([0,.5,1],float)
        elif a==2:
            avail=np.array([1,0,1])
            loss=np.array([1,0,0],float)
        elif a==3:
            avail=np.array([0,1,1])
            loss=np.array([0,0,1],float)

        eta = 1/(t+1)**0.5
        lam = 1/np.sqrt(t+1)

        choice[0], n, z = ucb(n, z, t, avail, loss)
        choice[1], L, p = SIEXP3(L, eta, avail, loss)
        choice[2], cum_loss, S = sleeping_exp3(cum_loss,S,eta,lam,avail,loss)

        for j in range(3):
            regret_ordering[j,i,t] = loss[choice[j]]
            regret_external_aux[j,:] += (loss[choice[j]]-loss)*avail
            regret_internal_aux[j,choice[j],:] += (loss[choice[j]]-loss)*avail
            regret_external[j,i,t] = np.max(regret_external_aux[j,:])
            regret_internal[j,i,t] = np.max(regret_internal_aux[j,:,:])
            
    for j in range(3):
        regret_ordering[j,i,:] = np.cumsum(regret_ordering[j,i,:])


sampling_rules = ["UCB","SI-EXP3","S-EXP3"]
regret_names = ["Policy", "External", "Internal"]
regrets = np.zeros((3,3,niter,T))
regrets[0,:,:,:] = regret_ordering
regrets[1,:,:,:] = regret_external
regrets[2,:,:,:] = regret_internal

T_plot = T
for k in range(3):
    fig=plt.figure(figsize=(4,3), dpi= 100, facecolor='w', edgecolor='k')  # to have big plots
    for i in range(3):
        means = np.mean(regrets[k,i,:,:], axis=0)
        quantiles = mstats.mquantiles(regrets[k,i,:,:], axis=0)
        plt.plot(np.arange(T_plot-1),means[1:T_plot], label=sampling_rules[i], color = colors[i+1], linewidth=2)
        plt.fill_between(np.arange(T_plot-1), quantiles[0,1:T_plot] ,quantiles[2,1:T_plot], color= colors[i+1], alpha=.2)
    plt.xlabel("Time")
    plt.ylabel(regret_names[k]+' Regret')
    plt.legend()
    plt.savefig('exp2_adversarial_'+regret_names[k]+'.pdf', bbox_inches = "tight")
    plt.show()


# ### Experiment 3
# 
# IID Bernoulli losses and availabilities
# This experiment reproduces the Stochastic environment experiment of Figure 2.


np.random.seed(3)
sampling_rules = ["UCB","SI-EXP3","S-EXP3"]

T = 5000
K = 10
niter = 20

regret_ordering = np.zeros((3,niter,T))
regret_external = np.zeros((3,niter,T))
regret_internal = np.zeros((3,niter,T))
choice = np.zeros(3,dtype=np.int8)

for i in tqdm(range(niter)):
    regret_external_aux = np.zeros((3,K))
    regret_internal_aux = np.zeros((3,K,K))
    mu=np.random.uniform(low=0,high=1,size=K)
    sleep=np.random.uniform(low=0,high=1,size=K)

    n=np.zeros(K)
    z=np.zeros(K)
    L=np.zeros((K,K))

    S = np.empty((0,K), int)
    cum_loss =np.zeros(K)
    
    for t in range(T):
        loss=np.array(np.random.binomial(n=1,p=mu,size=K),float)
        avail=np.array(np.random.binomial(n=1,p=sleep,size=K))
        while 1 not in avail:
            avail=np.array(np.random.binomial(n=1,p=sleep,size=K))

        eta = 1/(t+1)**0.5
        lam = 1/np.sqrt(t+1)

        choice[0], n, z = ucb(n, z, t, avail, loss)
        choice[1], L, p = SIEXP3(L, eta, avail, loss)
        choice[2], cum_loss, S = sleeping_exp3(cum_loss,S,eta,lam,avail,loss)
        
        idx = np.where(avail >0)
        oracle_choice = idx[0][np.argmin(mu[idx])]

        for j in range(3):
            regret_ordering[j,i,t] = loss[choice[j]]-loss[oracle_choice]
            regret_external_aux[j,:] += (loss[choice[j]]-loss)*avail
            regret_internal_aux[j,choice[j],:] += (loss[choice[j]]-loss)*avail
            regret_external[j,i,t] = np.max(regret_external_aux[j,:])
            regret_internal[j,i,t] = np.max(regret_internal_aux[j,:,:])
                
    for j in range(3):
        regret_ordering[j,i,:] = np.cumsum(regret_ordering[j,i,:])
        


regret_names = ["Policy", "External", "Internal"]
regrets = np.zeros((3,3,niter,T))
regrets[0,:,:,:] = regret_ordering
regrets[1,:,:,:] = regret_external
regrets[2,:,:,:] = regret_internal
N = 3
T_plot = T
for k in range(N):
    fig=plt.figure(figsize=(4,3), dpi= 100, facecolor='w', edgecolor='k')  # to have big plots
    for i in range(3):
        means = np.mean(regrets[k,i,:,:], axis=0)
        quantiles = mstats.mquantiles(regrets[k,i,:,:], axis=0)
        plt.plot(np.arange(T_plot-1),means[1:T_plot], label=sampling_rules[i], color = colors[i+1], linewidth=2)
        plt.fill_between(np.arange(T_plot-1), quantiles[0,1:T_plot] ,quantiles[2,1:T_plot], color= colors[i+1], alpha=.2)
    plt.xlabel("Time")
    plt.ylabel(regret_names[k]+' Regret')
    plt.legend()
    plt.savefig('exp3_full_iid_'+regret_names[k]+'.pdf', bbox_inches = "tight")


# ### Experiment 4
# Stochastic environment with dependence
# This experiment reproduces Figure 3.


np.random.seed(4)
sampling_rules = ["UCB","SI-EXP3","S-EXP3"]

T = 5000
K = 5
niter = 20

regret_ordering = np.zeros((3,niter,T))
regret_external = np.zeros((3,niter,T))
regret_internal = np.zeros((3,niter,T))
choice = np.zeros(3,dtype=np.int8)

M = 5
A = np.zeros((M,K), int)
mu = np.zeros((M,K), float)

for i in range(M):
    A[i]=np.random.binomial(n=1,p=.5,size=K)
    while 1 not in A[i]:
        A[i]=np.random.binomial(n=1,p=.5,size=K)
    mu[i]=np.random.uniform(low=0,high=1,size=K)
    
for i in tqdm(range(niter)):
    regret_external_aux = np.zeros((3,K))
    regret_internal_aux = np.zeros((3,K,K))

    n=np.zeros(K)
    z=np.zeros(K)
    L=np.zeros((K,K))

    S = np.empty((0,K), int)
    cum_loss =np.zeros(K)
    
    
    for t in range(T):
        a=random.randint(0,M-1)
        loss=np.array(np.random.binomial(n=1,p=mu[a],size=K),float)
        avail=A[a]

        eta = 1/(t+1)**0.5
        lam = 1/np.sqrt(t+1)

        choice[0], n, z = ucb(n, z, t, avail, loss)
        choice[1], L, p = SIEXP3(L, eta, avail, loss)
        choice[2], cum_loss, S = sleeping_exp3(cum_loss,S,eta,lam,avail,loss)
        
        idx = np.where(avail >0)
        oracle_choice = idx[0][np.argmin(mu[a,idx])]
        
        for j in range(3):
            regret_ordering[j,i,t] = loss[choice[j]]-loss[oracle_choice]
            regret_external_aux[j,:] += (loss[choice[j]]-loss)*avail
            regret_internal_aux[j,choice[j],:] += (loss[choice[j]]-loss)*avail
            regret_external[j,i,t] = np.max(regret_external_aux[j,:])
            regret_internal[j,i,t] = np.max(regret_internal_aux[j,:,:])
                
    for j in range(3):
        regret_ordering[j,i,:] = np.cumsum(regret_ordering[j,i,:])



regret_names = ["Policy", "External", "Internal"]
regrets = np.zeros((3,3,niter,T))
regrets[0,:,:,:] = regret_ordering
regrets[1,:,:,:] = regret_external
regrets[2,:,:,:] = regret_internal

T_plot = T
for k in range(N):
    fig=plt.figure(figsize=(4,3), dpi= 100, facecolor='w', edgecolor='k')  # to have big plots
    for i in range(3):
        means = np.mean(regrets[k,i,:,:], axis=0)
        quantiles = mstats.mquantiles(regrets[k,i,:,:], axis=0)
        plt.plot(np.arange(T_plot-1),means[1:T_plot], label=sampling_rules[i], color = colors[i+1], linewidth=2)
        plt.fill_between(np.arange(T_plot-1), quantiles[0,1:T_plot] ,quantiles[2,1:T_plot], color= colors[i+1], alpha=.2)
    plt.xlabel("Time")
    plt.ylabel(regret_names[k]+' Regret')
    plt.legend()
    plt.savefig('exp4_iid_dependent_'+regret_names[k]+'.pdf', bbox_inches = "tight")


# ### Exp 5: Rock Paper Scissors
# This experiment reproduces the "Rock, Paper, Scissors" environment of Figure 2


np.random.seed(5)
import nashpy as nash

sampling_rules = ["SIEXP3", "UCB", "SEXP3"]
action_names = ["Rock","Paper","Scissors"]

T = 500
K = 3
N = len(sampling_rules)
niter = 20
M = 4 # number of availability sets

regret_ordering = np.zeros((3,niter,T))
regret_external = np.zeros((3,niter,T))
regret_internal = np.zeros((3,niter,T))
choice = np.zeros(3,dtype=np.int8)


for i in tqdm(range(niter)):
    regret_external_aux = np.zeros((3,K))
    regret_internal_aux = np.zeros((3,K,K))
    weight = np.zeros((M,K,T))
    ta = np.zeros(M, int)
    n=np.zeros(K)
    z=np.zeros(K)
    L=np.zeros((K,K))

    S = np.empty((0,K), int)
    cum_loss =np.zeros(K)
    
    G = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]) # rock paper scisors matrix
    G = (G + 1)/2
    
    A = np.zeros((M,K), int)
    w = np.zeros((M,K), float)

    A[0] = [1, 1, 1]
    A[1] = [0, 1, 1]
    A[2] = [1, 0, 1]
    A[3] = [1, 1, 0]
    
    for j in range(M):
        k=sum(A[j])
        idx = np.array(np.where(A[j]>0)).reshape(k,1)
        G0 = G[idx,idx.T]
        rps = nash.Game(G0)
        w[j,idx.T] = list(rps.support_enumeration())[0][0]
    
    
    for t in range(T):
        a=np.random.randint(0,M)
        adv=np.random.choice(K,1,p=w[a]) # adversary plays Nash equilibrium
        loss=np.concatenate(1-G[:,adv])
        avail=A[a]
        
        eta = 1/(t+1)**0.5
        lam = 1/np.sqrt(t+1)


        choice[0], L, p = SIEXP3(L, eta, avail, loss)
        choice[1], n, z = ucb(n, z, t, avail, loss)
        
        weight[a,:,ta[a]] = p
        ta[a] += 1
            
        choice[2], cum_loss, S = sleeping_exp3(cum_loss,S,eta,lam,avail,loss)
        
        for j in range(3):
            regret_ordering[j,i,t] = loss[choice[j]]-.5
            regret_external_aux[j,:] += (loss[choice[j]]-loss)*avail
            regret_internal_aux[j,choice[j],:] += (loss[choice[j]]-loss)*avail
            regret_external[j,i,t] = np.max(regret_external_aux[j,:])
            regret_internal[j,i,t] = np.max(regret_internal_aux[j,:,:])
            
                
    for j in range(N):
        regret_ordering[j,i,:] = np.cumsum(regret_ordering[j,i,:])


regret_names = ["Policy", "External", "Internal"]
regrets = np.zeros((3,3,niter,T))
regrets[0,:,:,:] = regret_ordering
regrets[1,:,:,:] = regret_external
regrets[2,:,:,:] = regret_internal

T_plot = T
for k in range(N):
    fig=plt.figure(figsize=(4,3), dpi= 100, facecolor='w', edgecolor='k')  # to have big plots
    for i in range(3):
        means = np.mean(regrets[k,i,:,:], axis=0)
        quantiles = mstats.mquantiles(regrets[k,i,:,:], axis=0)
        plt.plot(np.arange(T_plot-1),means[1:T_plot], label=sampling_rules[i], color = colors[i+1], linewidth=2)
        plt.fill_between(np.arange(T_plot-1), quantiles[0,1:T_plot] ,quantiles[2,1:T_plot], color= colors[i+1], alpha=.2)
    plt.xlabel("Time")
    plt.ylabel(regret_names[k]+' Regret')
    plt.legend()
    plt.savefig('exp5_rock_paper_scissors_'+regret_names[k]+'.pdf', bbox_inches = "tight")
    


# ### Experiment 6
# Random zero-sum game with unavailable actions for both players 
# This experiment reproduces Figure 4

np.random.seed(6)
sampling_rules = ["SIEXP3", "UCB", "SEXP3"]
action_names = ["Rock","Paper","Scissors"]

T = 5000
K = 5
N = len(sampling_rules)
niter = 20
M = 4 # number of availability sets

regret_ordering = np.zeros((3,niter,T))
regret_external = np.zeros((3,niter,T))
regret_internal = np.zeros((3,niter,T))
choice = np.zeros(3,dtype=np.int8)

for i in tqdm(range(niter)):
    regret_external_aux = np.zeros((3,K))
    regret_internal_aux = np.zeros((3,K,K))
    weight = np.zeros((M,K,T))
    ta = np.zeros(M, int)
    n=np.zeros(K)
    z=np.zeros(K)
    L=np.zeros((K,K))

    S = np.empty((0,K), int)
    cum_loss =np.zeros(K)
    
    G = np.random.uniform(low=0, high=1, size=(K, K))
    G = (np.triu(G) + np.triu(1-G).T) - np.identity(K) / 2


    A = np.zeros((M,K), int)
    w = np.zeros((M,K), float)

    for j in range(M):
        A[j]=np.random.binomial(n=1,p=.5,size=K)
        while sum(A[j]) < 2:
            A[j]=np.random.binomial(n=1,p=.7,size=K)
        
        k=sum(A[j])
        idx = np.array(np.where(A[j]>0)).reshape(k,1)
        G0 = G[idx,idx.T]
        rps = nash.Game(G0)
        w[j,idx.T] = list(rps.support_enumeration())[0][0]
    
    
    for t in range(T):
        a=np.random.randint(0,M)
        adv=np.random.choice(K,1,p=w[a]) # adversary plays Nash equilibrium
        loss=np.concatenate(1-G[:,adv])
        avail=A[a]
        
        eta = 1/(t+1)**0.5
        lam = 1/np.sqrt(t+1)


        choice[0], L, p = SIEXP3(L, eta, avail, loss)
        choice[1], n, z = ucb(n, z, t, avail, loss)
        
        weight[a,:,ta[a]] = p
        ta[a] += 1
            
        choice[2], cum_loss, S = sleeping_exp3(cum_loss,S,eta,lam,avail,loss)
        
        for j in range(3):
            regret_ordering[j,i,t] = loss[choice[j]]-.5
            regret_external_aux[j,:] += (loss[choice[j]]-loss)*avail
            regret_internal_aux[j,choice[j],:] += (loss[choice[j]]-loss)*avail
            regret_external[j,i,t] = np.max(regret_external_aux[j,:])
            regret_internal[j,i,t] = np.max(regret_internal_aux[j,:,:])
    
    for j in range(N):
        regret_ordering[j,i,:] = np.cumsum(regret_ordering[j,i,:])


regret_names = ["Policy", "External", "Internal"]
regrets = np.zeros((3,3,niter,T))
regrets[0,:,:,:] = regret_ordering
regrets[1,:,:,:] = regret_external
regrets[2,:,:,:] = regret_internal

T_plot = T
for k in range(N):
    fig=plt.figure(figsize=(4,3), dpi= 100, facecolor='w', edgecolor='k')  # to have big plots
    for i in range(3):
        means = np.mean(regrets[k,i,:,:], axis=0)
        quantiles = mstats.mquantiles(regrets[k,i,:,:], axis=0)
        plt.plot(np.arange(T_plot-1),means[1:T_plot], label=sampling_rules[i], color = colors[i+1], linewidth=2)
        plt.fill_between(np.arange(T_plot-1), quantiles[0,1:T_plot] ,quantiles[2,1:T_plot], color= colors[i+1], alpha=.2)
    plt.xlabel("Time")
    plt.ylabel(regret_names[k]+' Regret')
    plt.legend()
    plt.savefig('exp6_random_zero_sum_game_'+regret_names[k]+'.pdf', bbox_inches = "tight")

