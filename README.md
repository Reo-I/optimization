# optimization
Minimize the following problem:
$$
\underset{\bm{w}\in\mathbb{R}^n}{min} f(\bm{w}):=||\bm{b} - \bm{A}\bm{w}||^2 + \lambda||\bm{w}||^2, 
$$
where $\bm{A}\in\mathbb{R}^{m\times n}, \bm{b}\in\mathbb{R}^m, \lambda \in \mathbb{R}, m<n$.

First of all, for the problem setting, $\bm{A}, \bm{w}$ are randomly generated, and using these generated matrix, $\bm{b}$ is calcurated as follows:

$$
\bm{b}:=\bm{A}\bm{w} + \bm{\epsilon},　\epsilon\sim \mathcal{N}(0, 1^2),  
$$

# Usage
 
## optimize the $f(\bm{b})$ 
 
```bash
$ git clone git@github.com:Reo-I/optimization.git
$ cd optimization
$ python minimize.py 
```