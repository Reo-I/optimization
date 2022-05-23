import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--solve_type', type=int, default='0')
args = parser.parse_args()

class Minimization:
    def __init__(self, m=10, n=20, lams = [0,  0.1,1,  10], opt_type = ["l_smooth"]*4, num_iter = 30):
        self.m = m
        self.n = n
        A = np.random.randn(m, n-1)
        self.A = np.concatenate([np.ones((self.m, 1)), A] ,axis = 1) # num data * input data
        w = np.random.randn(n, 1)
        self.b =np.dot(self.A, w) + np.random.normal(0, 1., (self.m, 1)) #target data
                                                        
        self.lams = lams
        self.xi = 0.3
        self.f_each = {}
        self.num_iter = num_iter
        self.opt_types = opt_type
    

    def f(self, w, lam):
        # return ||b - Aw||^2 + lambda | w |^2
        d = self.b - np.dot(self.A, w)
        d_2 = np.dot(d.T, d)
        return (d_2 + lam * np.dot(w.T,w)).flatten()[0].real
    
    def grad_f(self, w, lam):
        return 2*(np.dot(np.dot(self.A.T, self.A), w) - np.dot(self.A.T, self.b) + lam*w)
    
    def step_size_L(self, i):
        #alpha = 1/L  (L:固有値の最大値)
        hesse = 2*(np.dot(self.A.T, self.A) + self.lams[i] * np.eye(self.n))
        e, v = np.linalg.eig(hesse)
        return 1 / max(e)
    
    def step_size_armijo(self, w_k, grad, i):
        # decide step size based on armijo's rule
        alpha = 1
        while self.f(w_k - alpha*grad,  self.lams[i]) > self.f(w_k,  self.lams[i]) - self.xi*alpha*np.dot(grad.T, grad):
            alpha = 0.9*alpha
        return alpha
    
    def minize(self):
        init_w = np.random.randn(self.n, 1)
        for i in range(len(self.lams)):
            f_k = [0]*self.num_iter
            w_k = init_w.copy()
            w_k1 = init_w.copy()
            
            if (self.opt_types[i] == "l_smooth") or (self.opt_types[i] == "nesterov"):
                # if type is L smooth, the alpha is decisive
                alpha = self.step_size_L(i)

            for k in range(self.num_iter):
                grad = self.grad_f(w_k, self.lams[i])
                
                if self.opt_typeｓ[i] == "armijo":
                    alpha = self.step_size_armijo(w_k, grad, i)
                    
                #update w
                if self.opt_types[i] == "l_smooth" or self.opt_types[i] == "armijo" :
                    w_k = w_k - alpha *grad
                
                elif self.opt_types[i] == "nesterov":
                    beta = (k)/(k+3)
                    y_k = w_k + beta * (w_k - w_k1)
                    w_k1 = w_k.copy()
                    w_k = y_k - alpha *self. grad_f(y_k, self.lams[i])
                
                f_k[k] = self.f(w_k, self.lams[i])
            self.f_each[(self.opt_types[i], self.lams[i])] = f_k
    
    def visualize(self, save_path):
        for i in range(len(self.lams)):
            if self.opt_types[i] == "l_smooth":
                l_style = "solid"
            else:
                l_style = "dashed"
            plt.plot(range(self.num_iter), self.f_each[(self.opt_types[i], self.lams[i])], \
                     linestyle=l_style, label = f"opt:{self.opt_types[i]}, lambda : {self.lams[i]}")
        plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("f(w)")
        plt.title(f"{save_path}")
        plt.savefig(f"img/{save_path}.png")
        plt.ylim([0, 30])
        plt.savefig(f"img/{save_path}_small.png")
        plt.show()

if __name__ == "__main__":
    if args.solve_type == 0:
        #L-smooth, optimization
        do_min = Minimization( lams = [0, 0.1, 1, 10], opt_type=["l_smooth"]*4, num_iter = 30)
        do_min.minize()
        do_min.visualize(save_path = "q1_steepest_descent")


    elif args.solve_type == 1:
        #armijo's rule
        do_min = Minimization( lams = [0.5, 0.5, 1, 1], opt_type=["l_smooth", "armijo"]*2, num_iter = 30)
        do_min.minize()
        do_min.visualize(save_path = "q2_L_smooth_armijo")
    

    elif args.solve_type == 2:
        #Nesterov's accelerated gradient algorithm
        do_min = Minimization( lams = [0.5, 0.5,  2, 2], opt_type=["l_smooth", "nesterov"]*2, num_iter = 60)
        do_min.minize()
        do_min.visualize(save_path = "q3_L_smooth_nesterov")
