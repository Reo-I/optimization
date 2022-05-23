# optimization
Minimize the following problem:



$$
\underset{\mathbf{w}\in\mathbb{R}^n}{min} f(\mathbf{w}) := || \mathbf{b} - \mathbf{A}\mathbf{w}||^2 + \lambda||\mathbf{w}||^2
$$

where $\mathbf{A}\in\mathbb{R}^{m\times n}, \mathbf{b}\in\mathbb{R}^m, \lambda \in \mathbb{R}$ , and $m　<　n$ .

First of all, for the problem setting, $\mathbf{A}, \mathbf{w}$ are randomly generated, and using these generated matrix, $\mathbf{b}$ is calcurated as follows:

$$
\mathbf{b}:=\mathbf{A}\mathbf{w} + \mathbf{\epsilon},　\epsilon\sim \mathcal{N}(0, 1^2),  
$$

# Usage
 
optimize the $f(\mathbf{w})$ 
 
```bash
$ git clone git@github.com:Reo-I/optimization.git
$ cd optimization
$ python minimize.py --solve_type 0
```

Command line option `--solve_type` is the options to devide how to solve this problem.

`--solve_type 0` means that optimize $f(\mathbf{w})$ using step size $\alpha=\frac{1}{L}$. $L$ represents 
the maximum eigenvalues ​​of the Hessian matrix. 

`--solve_type 1 `  means that optimize $f(\mathbf{w})$ searching step size by Armijo rule. 

`--solve_type 2 `  means that optimize $f(\mathbf{w})$ using the Nesterov's accelerated gradient algorithm. 
