# -*- coding: utf-8 -*-

# Implements Xiu et. al. (2019):
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3389884
import numpy as np
from scipy.optimize import minimize

#%%
def screen_words(vocab,
                 doc_term_mat,
                 returns,
                 alpha_plus=0.1,
                 alpha_minus=0.1,
                 kappa=0.5):
    """
    Screens for sentiment-charged words. Gives a wordlist S_hat.
    
    Returns:
        Indices of the words' columns in the doc_term matrix
    """
    
    X_binary = np.where(doc_term_mat > 0, 1, 0)
    y_binary = np.where(returns > 0, 1, 0).reshape(-1,1)
    
    k = np.sum(X_binary, axis=0)
    f = np.sum(X_binary * y_binary, axis=0) / k

    ind_plus = np.where( f.flatten() > 0.5+alpha_plus )
    ind_minus = np.where( f.flatten() < 0.5-alpha_minus )
    
    ind = np.concatenate((ind_plus, ind_minus))
    ind_kappa = np.where( k>=kappa )
    
    S_hat = ind[np.isin(ind, ind_kappa)].sort()
    
    return S_hat
    
#%%
def learn_sentiment_topics(y, doc_term_mat, S_hat):
    if np.ndim(y) != 1:
        raise ValueError('y should be 1d array')
    order = y.argsort()
    ranks = order.argsort()+1
    p_hat = ranks/len(y)
    
    D_s_hat = doc_term_mat[:, S_hat]
    s_hat = np.sum(D_s_hat, axis=1)
    
    D_hat = (D_s_hat / s_hat.reshape(-1,1)).T
    W_hat = np.stack((p_hat, 1-p_hat))
    
    O_hat = D_hat @ W_hat.T @ np.linalg.inv(W_hat @ W_hat.T)
    O_hat = O_hat.clip(min=0)
    O_hat = O_hat/np.linalg.norm(O_hat, ord=1, axis=0)
    
    return O_hat
    
#%%
def log_likelihood(s_hat, d, p, O_hat, lamb):
    assert type(s_hat)==int or type(s_hat)==float
    assert d.shape[0]==O_hat.shape[0]
    assert O_hat.shape[1]==2
    l_l = (1/s_hat)*(d*np.log(p*O_hat[:,0] + (1-p)*O_hat[:,1])).sum() + lamb*np.log(p*(1-p))
    return l_l

def log_likelihood_of_p(p):
    return 1 # log_likelihood(s_hat)



class SentimentExtractionViaScreeningAndTopicModeling():
    def __init__(self,
        alpha_plus=0.1,
        alpha_minus=0.1,
        kappa=0.5):
    self.alpha_plus = alpha_plus
    self.alpha_minus = alpha_minus
    self.kappa = kappa
    self.p = 0.


    def _screen_words(self,
        doc_term_mat,
        returns,
        alpha_plus=0.1,
        alpha_minus=0.1,
        kappa=0.5):
        """
        Screens for sentiment-charged words. Gives a wordlist S_hat.
        
        Returns:
            Indices of the words' columns in the doc_term matrix
        """
        
        X_binary = np.where(doc_term_mat > 0, 1, 0)
        y_binary = np.where(returns > 0, 1, 0).reshape(-1,1)
        
        k = np.sum(X_binary, axis=0)
        f = np.sum(X_binary * y_binary, axis=0) / k

        ind_plus = np.where( f.flatten() > 0.5+alpha_plus )
        ind_minus = np.where( f.flatten() < 0.5-alpha_minus )
        
        ind = np.concatenate((ind_plus, ind_minus))
        ind_kappa = np.where( k>=kappa )
        
        S_hat = ind[np.isin(ind, ind_kappa)].sort()
        
        return S_hat


    def _learn_sentiment_topics(self, y, doc_term_mat, S_hat):
        if np.ndim(y) != 1:
            raise ValueError('y should be 1d array')
        order = y.argsort()
        ranks = order.argsort()+1
        p_hat = ranks/len(y)
        
        D_s_hat = doc_term_mat[:, S_hat]
        s_hat = np.sum(D_s_hat, axis=1)
        
        D_hat = (D_s_hat / s_hat.reshape(-1,1)).T
        W_hat = np.stack((p_hat, 1-p_hat))
        
        O_hat = D_hat @ W_hat.T @ np.linalg.inv(W_hat @ W_hat.T)
        O_hat = O_hat.clip(min=0)
        O_hat = O_hat/np.linalg.norm(O_hat, ord=1, axis=0)
        
        return O_hat

    def _log_likelihood(self, p, s_hat, d, O_hat, lamb):
        assert type(s_hat)==int or type(s_hat)==float
        assert d.shape[0]==O_hat.shape[0]
        assert O_hat.shape[1]==2
        l_l = (1/s_hat)*(d*np.log(p*O_hat[:,0] + (1-p)*O_hat[:,1])).sum() + lamb*np.log(p*(1-p))
        return l_l


    def _arg_maximize(self, d, p, O_hat, lamb):        
        # def _log_likelihood_of_p(self, p):
        #     return self._log_likelihood(self, s_hat, d, p, O_hat, lamb):

        # x0 = np.array([0.5])
        # res = minimize(rosen, x0, method='nelder-mead',
        # options={'xtol': 1e-8, 'disp': True})
        f = lambda p: self._log_likelihood(p, s_hat, d, O_hat, lamb)
        
        bnds = (0,1)
        
        result = minimize(fun=self._log_likelihood,
                          x0=0.5, 
                          args=(s_hat, d, O_hat, lamb),
                          method='SLSQP',
                          bounds = bnds)
        
        return result # TODO


    def fit(self, X, y=None):
        """Learn SESTM model

        Parameters
        ----------
        X: ndarray, shape=(n_samples, n_features)
           Document word matrix.

        y: ndarray, shape=(n_samples,)
           Stock returns

        Returns
        -------
        self
        """

    def transform(self, X):
        return self.p # TODO


#%%
if __name__ == "__main__":
    print('nothing to run')