#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import itertools
from scipy.stats import norm


# In[2]:


class HMM:
    def __init__(self,A,B,pi):
        self.A = A 
        self.B = B
        self.pi = pi
   
    #向前算法（F为/alpha_t(i)）
    def _forward(self,obs_seq):
        #取A = N x N
        N = self.A.shape[0]
        T = len(obs_seq)
    
        F = np.zeros((N,T))
        
        #alpha = pi*b
        F[:,0] = self.pi * self.B[:,obs_seq[0]]
        
        for t in range(1,T):
            for n in range(N):
                #计算第t个时，第n个状态的前向概率
                F[n,t] = np.dot(F[:,t-1],(self.A[:,n]))*self.B[n,obs_seq[t]]
        return F

    #向后算法
    def _backward(self,obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)
        
        X = np.zeros((N,T))
        #表示X矩阵的最后一列
        X[:,-1:] = 1

        for t in reversed(range(T-1)):
            for n in range(N):
                # 边权值为a_ji
                X[n,t] = sum(X[:,t+1]*self.A[n,:]*self.B[:,obs_seq[t+1]])
        
        return X
    
    #维特比算法
    def viterbi(self,obs_seq):
        
        N = self.A.shape[0]
        T = len(obs_seq)
        
        prev = np.zeros((T-1,N),dtype = int)
        
        V = np.zeros((N,T))
        V[:,0]=self.pi*self.B[:,obs_seq[0]]
        
        for t in range(1,T):
            for n in range(N):
                #计算delta(j)*a_ji
                seq_probs = V[:,t-1]*self.A[:,n]*self.B[n,obs_seq[t]]
                #计算最大状态转移过程
                prev[t-1,n] = np.argmax(seq_probs)
                V[n,t] = max(seq_probs)
                
        return V,prev
    
    def build_viterbi_path(self,prev,last_state):
        """
        returns a state path ending in last_state in reverse order
        """
        T = len(prev)
        yield(last_state)

        #从T-1开始，每次下降1
        for i in range(T-1,-1,-1):
            yield(prev[i,last_state])
            last_state = prev[i,last_state]

    def stat_path(self,obs_seq):
        V,prev = self.viterbi(obs_seq)
        
        #build state path with greatest probability
        last_state = np.argmax(V[:,-1]) 
        path = list(self.build_viterbi_path(prev,last_state))
        
        return V[last_state,-1],reversed(path)
    
    #Baum-Welch算法
    def baum_welch_train(self,observations,criterion=0.05):
        n_states = self.A.shape[0]
        #观察序列长度T
        n_samples = len(observations)
        
        done = False
        while not done:
            #alpha_t(i) = P(o_1,o_2,...o_t,q_t = s_i | hmm)
            #Initialize alpha
            #获得所有向前传播节点值alpha_t(i)
            alpha = self._forward(observations)
            
            
            # beta_t(i) = P(o_t+1,o_t+2,...o_T | q_t = s_i, hmm)
            # Initialize beta
            # 获得所有后向传播节点值
            beta = self._backward(observations)
            
            # compute xi_t(i,j) -> xi(i,j,t)
            xi = np.zeros((n_states,n_states,n_samples-1))
            # in each moment
            for t in range(n_samples-1):
                # compute P(O|hmm)
                denom = sum(alpha[:,-1])
                for i in range(n_states):
                    # numer[1,:] = row vector, alpha[i,t] = real number, self.A[i,:] = column vector
                    # self.B[:,observations[t+1].T] = row vector, beta[:,t+1].T = column vector
                    numer = alpha[i,t] * self.B[:,observations[t+1]].T * beta[:,t+1].T
                    xi[i,:,t] = numer/denom
                    
            # compute gamma_t(i) sum the j values
            gamma = np.sum(xi,axis = 1)
            # need final gamma elements for new B
            prod = (alpha[:,n_samples - 1] * beta[:,n_samples-1]).reshape((-1,1))
            # sum the nodes that all from T time
            gamma = np.hstack((gamma, prod / np.sum(prod)))
            #colum vector
            newpi = gamma[:,0]
            newA = np.sum(xi,2) / np.sum(gamma[:,:-1], axis = 1).reshape((-1,1))
            newB = np.copy(self.B) 

            # the observation state number
            num_levels = self.B.shape[1]
            summgamma = np.sum(gamma, axis = 1)
            for lev in range(num_levels):
                mask = observations == lev
                newB[:,lev] = np.sum(gamma[:,mask],axis =1) / summgamma

            if np.max(abs(self.pi - newpi)) < criterion and \
               np.max(abs(self.A - newA)) < criterion and \
               np.max(abs(self.B - newB)) < criterion:
                done = 1

            self.A[:],self.B[:],self.pi[:] = newA,newB,newpi

    #模拟序列生成函数
    def simulate(self,T):
        def draw_from(probs):
            #np.random.multinomial 为多项式分布，1为试验次数，probs是每个点数的概率，均为1/6
            #给定行向量的概率，投掷次数为1次，寻找投掷的点数
            return np.where(np.random.multinomial(1,probs)==1)[0][0]
        
        observations = np.zeros(T,dtype = int)
        states = np.zeros(T,dtype = int)
        states[0] = draw_from(self.pi)
        observations[0] = draw_from(self.B[states[0],:])
        for t in range(1,T):
            states[t] = draw_from(self.A[states[t-1],:])
            observations[t] = draw_from(self.B[states[t],:])
        return observations,states


# In[3]:


class HMM_spike:
    def __init__(self,mu,P,cov):
        self.P = P #spike shoting rate 神经元发放速率
        self.mu = mu
        self.cov = cov
        
    # 计算 probability matrix A 状态转移概率矩阵 K * K
    def proba_matrix(self,k):
        tran_pro = np.append(np.zeros([1,k]),np.append(np.eye(k-1,dtype=float),np.zeros([k-1,1]),axis=1),axis=0)
        tran_pro[0,0],tran_pro[1,0],tran_pro[0,-1] = 1-self.P,self.P,1
        
        return tran_pro
    
    #计算 state-conditional sample probability B 观察概率 K * T
    def state_conditional(self,obs_seq,K):
        T = obs_seq.shape[0]
        U = self.mu * np.ones((1,T))
        st_con_mat = np.exp(-1 * np.
                            square(np.ones((K,1)) * obs_seq - U ) / (2 * np.square(self.cov))) / ((2 * np.pi) ** 0.5 * self.cov)

        return st_con_mat
        
    #向前算法（F为/alpha_t(i)）
    def _forward(self,K,T):
        #取A = N x N
        F = np.mat(np.zeros((K,T)))
        
        #alpha = pi*b
        F[:,0] = 1
        for t in range(1,T):
            F[:,t] = np.multiply(self.B[:,t],self.A.T * F[:,t-1])
            #rescaled
            F[:,t] = F[:,t] / np.linalg.norm(F[:,t],ord=1)
        
        return F    

    #向后算法
    def _backward(self,K,T):
        X = np.mat(np.zeros((K,T)))
        #表示X矩阵的最后一列
        X[:,-1:] = 1
        
        for t in reversed(range(T-1)):
            # 边权值为a_ji
            X[:,t] = self.A * np.multiply(self.B[:,t+1], X[:,t+1])
            #rescaled
            X[:,t] = X[:,t] / np.linalg.norm(X[:,t],ord=1)
        return X
    
    #维特比算法
    def viterbi(self,obs_seq):
        
        N = self.A.shape[0]
        T = len(obs_seq)
        
        prev = np.zeros((T-1,N),dtype = int)
        
        V = np.zeros((N,T))
        V[0,0] = 1
        
        for t in range(1,T):
            for n in range(N):
                #计算delta(j)*a_ji
                seq_probs = V[:,t-1]*self.A[:,n]*self.B[n,t]
                #计算最大状态转移过程
                prev[t-1,n] = np.argmax(seq_probs)
                V[n,t] = max(seq_probs) 
        return V,prev
    
    def build_viterbi_path(self,prev,last_state):
        """
        returns a state path ending in last_state in reverse order
        """
        T = len(prev)
        yield(last_state)

        #从T-1开始，每次下降1
        for i in range(T-1,-1,-1):
            yield(prev[i,last_state])
            last_state = prev[i,last_state]

    def stat_path(self,obs_seq):
        V,prev = self.viterbi(obs_seq)
        
        #build state path with greatest probability
        last_state = np.argmax(V[:,-1]) 
        path = list(self.build_viterbi_path(prev,last_state))
        
        return V[last_state,-1],reversed(path)
    
    #Baum-Welch算法 for spike shooting
    def baum_welch_train(self,observations,criterion=0.05):
        n_states = self.mu.shape[0]
        #观察序列长度T
        n_samples = len(observations)
        
        self.A = self.proba_matrix(n_states)
        self.B = self.state_conditional(observations,n_states) 
        lock = 0
        while (lock < 8):
            #alpha_t(i) = P(o_1,o_2,...o_t,q_t = s_i | hmm)
            #Initialize alpha
            #获得所有向前传播节点值alpha_t(i)
            alpha = self._forward(n_states,n_samples)
            
            # beta_t(i) = P(o_t+1,o_t+2,...o_T | q_t = s_i, hmm) 
            # Initialize beta
            # 获得所有后向传播节点值
            beta = self._backward(n_states,n_samples)
            
            #gamma = np.multiply(alpha,beta)
            gamma = np.multiply(self.A.T * alpha,np.multiply(self.B ,beta))
            newmu = np.zeros((1,n_states))
             # in each moment
            for t in range(n_samples):
                gamma[:,t] = gamma[:,t] / np.linalg.norm(gamma[:,t], ord=1)
                newmu = np.dot(np.dot(observations[t], gamma[:,t].T).reshape(1,-1),np.linalg.pinv(np.dot(gamma[:,t].reshape(-1,1),gamma[:,t].reshape(1,-1))))
            #print(gamma)
        
            
            newcov = np.mean(np.multiply(observations,observations)) - np.mean(np.multiply(np.dot(newmu,gamma),observations))
            newP = np.sum(gamma[1,1:])/np.sum(gamma[0,:-1])
            
            if newP * 240000 < 0.5:
                break
            
            self.mu,self.cov,self.P = newmu.reshape(-1,1),newcov,newP
          #  print(newmu)
            
            self.A = self.proba_matrix(n_states)
            self.B = self.state_conditional(observations,n_states) 
            lock += 1
           
           
        return 


# In[4]:


class HMM_multi_spike:
    def __init__(self,mu,P,cov):
        self.P = P #spike shoting rate 神经元发放速率
        self.mu = np.mat(mu)
        self.cov = cov
    
    # 计算简化过的隐藏层层数
    def hidden_state(self,N,R,K):
    #allow no more than R (for example R = 2) neurons to simultaneously spike: K^N reduce to 1 + N(K-1) + N(N-1)(K-1)^2 / 2 
        if R > 3:
            R = 3
            print("no more than 3 simultaneously sipke neurons allowed")
        hk = 1
        HK = 1
        for i in range(1,R+1):
            hk = hk * (N - i+1) * ( K - 1 )
            HK += hk / i
        return int(HK)
    
    # 计算 probability matrix A 状态转移概率矩阵 K * K
    def proba_matrix(self,TotalK,K,N):
        #正常用np.kron
        tran_pro = np.append(np.zeros([1,TotalK]),np.append(np.eye(TotalK-1,dtype=float),np.zeros([TotalK-1,1]),axis=1),axis=0)
        for n in range(N):
            tran_pro[0,n*(K-1)],tran_pro[n*(K-1)+1,n*(K-1)]= 1,0
            tran_pro[n*(K-1)+1,0] = self.P[n]
        tran_pro[0,0],tran_pro[0,-1]= 1-sum(self.P),1
        return tran_pro
    
    #计算 state-conditional sample probability B 观察概率 K * T
    def state_conditional(self,TotalK,obs_seq,K,N):
        multi_mu_sum = np.mat(np.zeros((TotalK,obs_seq.shape[0])))
        multi_mu_sum[0,:] = norm.pdf(obs_seq,sum(self.mu[:,0]),self.cov**0.5)
        for n in range(N):
            for k in range(K-1):
                multi_mu_sum[1 + n*(K-1) + k,:] = norm.pdf(obs_seq,self.mu[k+1,n],self.cov**0.5)
        return multi_mu_sum
        
    #向前算法（F为/alpha_t(i)）
    def _forward(self,K,T):
        #取A = N x N
        
        F = np.mat(np.zeros((K,T)))
        
        #alpha = pi*b
        F[:,0] = 1
        for t in range(1,T):
            F[:,t] = np.multiply( self.B[:,t], np.sum(np.multiply(F[:,t-1],np.mat(self.A)),axis=1))
            #rescaled
            F[:,t] = F[:,t] / np.linalg.norm(F[:,t],ord=1)

        return F    

    #向后算法
    def _backward(self,K,T):
        
        X = np.mat(np.zeros((K,T)))
        #表示X矩阵的最后一列
        X[:,-1:] = 1
        
        for t in reversed(range(T-1)):
            # 边权值为a_ji
            # X[:,t] =  np.multiply( X[:,t+1], np.sum(np.multiply(self.B[:,t+1],np.mat(self.A).T),axis=0).T )
            X[:,t] =  np.sum(np.multiply( np.mat(self.A).T, np.multiply(self.B[:,t+1],X[:,t+1]) ),axis=0).T
            #rescaled
            X[:,t] = X[:,t] / np.linalg.norm(X[:,t],ord=1)
        return X
    
    #维特比算法
    def viterbi(self,obs_seq):
        
        N = self.A.shape[0]
        T = len(obs_seq)
        
        prev = np.zeros((T-1,N),dtype = int)
        
        V = np.zeros((N,T))
        V[0,0] = 1
        
        for t in range(1,T):
            for n in range(N):
                #计算delta(j)*a_ji
                seq_probs = V[:,t-1]*self.A[:,n]*self.B[n,t]
                #计算最大状态转移过程
                prev[t-1,n] = np.argmax(seq_probs)
                V[n,t] = max(seq_probs)
                
        return V,prev
    
    def build_viterbi_path(self,prev,last_state):
        """
        returns a state path ending in last_state in reverse order
        """
        T = len(prev)
        yield(last_state)

        #从T-1开始，每次下降1
        for i in range(T-1,-1,-1):
            yield(prev[i,last_state])
            last_state = prev[i,last_state]

    def stat_path(self,obs_seq):
        V,prev = self.viterbi(obs_seq)
        
        #build state path with greatest probability
        last_state = np.argmax(V[:,-1]) 
        path = list(self.build_viterbi_path(prev,last_state))
        
        return V[last_state,-1],reversed(path)
    
    #Baum-Welch算法 for spike shooting
    def baum_welch_train(self,observations,mspike = 1,maxterion=2):
        n_states = self.mu.shape[0] 
        n_neuron = self.mu.shape[1]
        #观察序列长度T
        n_samples = len(observations)
       
        n_totlK = self.hidden_state(n_neuron,mspike,n_states)
        state_name = list(range(0,n_states))
        #combin = list(itertools.product(state_name,repeat=n_neuron))

        self.A = self.proba_matrix(n_totlK,n_states,n_neuron)
        self.B = self.state_conditional(n_totlK,observations,n_states,n_neuron) 
        lock = 0
        while (lock < maxterion):
            #alpha_t(i) = P(o_1,o_2,...o_t,q_t = s_i | hmm)
            #Initialize alpha
            #获得所有向前传播节点值alpha_t(i)
            alpha = self._forward(n_totlK,n_samples)
            self.alpha = alpha
            # beta_t(i) = P(o_t+1,o_t+2,...o_T | q_t = s_i, hmm) 
            # Initialize beta
            # 获得所有后向传播节点值
            beta = self._backward(n_totlK,n_samples)
            self.beta = beta
            
            self.gamma = np.multiply(alpha,beta)  
            self.gamma = self.gamma / (np.linalg.norm(self.gamma,ord=1,axis=0))
                             
            self.titaa = np.zeros((n_neuron,n_states,n_samples))
            for n in range(n_neuron):
                self.titaa[n,0,:],self.titaa[n,1:,:] = self.gamma[0,:],self.gamma[n*(n_states-1) + 1:(n+1)*(n_states-1)+1,:]
            
            #update the P
            newP = np.zeros(n_neuron)
            for n in range(n_neuron):
                newP[n] = np.sum(self.titaa[n,1,1:])/np.sum(self.titaa[n,0,:-2])
            #if some ith probability p^i corresponded to f firing rate of 0.5hz or less
            newP = [0.99 * np.random.choice([x for x in newP if x > p])if ( p * 24000 <= 0.5 and p != max(newP) )else p for p in newP]
            #if signal-tp-onise variance of recording is very low --- noise covariance self.cov is very high
            #if newcov < 1:
            #    self.P = [10**(-10) for p in self.P]
            
            
            #update the mu
            siggma = np.mat(self.titaa.reshape((n_neuron * n_states ,n_samples)))
            newmu = np.dot(np.sum(np.multiply(observations,siggma),axis = 1).T,np.linalg.pinv(siggma * siggma.T))
            multi_mu = np.mat(newmu.reshape((n_neuron,-1)))
            
            #update the cov
            mu_sum = np.zeros((1,n_samples))
            for n in range(n_neuron):
                    mu_sum[0,:] = mu_sum[0,:] + multi_mu[n,:] * np.mat(self.titaa[n,:,:])
            newcov = np.mean(np.multiply(observations,observations)) - np.mean(np.multiply(mu_sum,observations))
            #print(np.max(abs(self.mu - newmu)),  np.max(abs(self.P - newP)), np.max(abs(self.cov - newcov))) 
            
            self.mu, self.cov, self.P = multi_mu.T, newcov, newP
            print(self.P)
            self.A = self.proba_matrix(n_totlK,n_states,n_neuron)
            self.B = self.state_conditional(n_totlK,observations,n_states,n_neuron) 
            lock += 1
           
           
        return 
    

