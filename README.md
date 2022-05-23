# optimization
Minimize the following problem:
$$
\underset{\boldsymbol{w}\in\mathbb{R}^n}{min} f(\boldsymbol{w}):=||\boldsymbol{b} - \boldsymbol{A}\boldsymbol{w}||^2 + \lambda||\boldsymbol{w}||^2, 
$$
where $\boldsymbol{A}\in\mathbb{R}^{m\times n}, \boldsymbol{b}\in\mathbb{R}^m, \lambda \in \mathbb{R}, m<n$.

First of all, for the problem setting, $\boldsymbol{A}, \boldsymbol{w}$ are randomly generated, and using these generated matrix, $\boldsymbol{b}$ is calcurated as follows:

$$
\boldsymbol{b}:=\boldsymbol{A}\boldsymbol{w} + \boldsymbol{\epsilon},　\epsilon\sim \mathcal{N}(0, 1^2),  
$$

# Usage
 
### optimize the $f(\boldsymbol{w})$ 
 
```bash
$ git clone git@github.com:Reo-I/optimization.git
$ cd optimization
$ python minimize.py --solve_type 0
```

Command line option `--solve_type` is the options to devide how to solve this problem.

`--solve_type 0` means that optimize $f(\boldsymbol{w})$ using step size $\alpha=\frac{1}{L}$. $L$ represents 
the maximum eigenvalues ​​of the Hessian matrix. 

`--solve_type 1 `  means that optimize $f(\boldsymbol{w})$ searching step size by Armijo rule. 

`--solve_type 2 `  means that optimize $f(\boldsymbol{w})$ using the Nesterov's accelerated gradient algorithm. 
