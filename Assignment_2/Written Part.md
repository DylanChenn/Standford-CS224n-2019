# Assignment 2: word2vec



### 1. Written: Understanding word2vec



(a) True output vector $y$ is 1 in $o^{th}$ position and 0 else where. So the cross-entropy $-\sum_{w\in Vocab}y_w\log(\hat{y}_w) = -\log(\hat{y}_o)$

(b) $J_{navie-softmax}(v_c,o,U)=-u_o^Tv_c+\log \sum_{w\in Vocab}\exp(u_w^Tv_c)$
$$
\begin{split}
\frac{\part J}{\part v_c} &= -u_o + \frac{\frac{\part}{\part v_c}\sum_{w\in Vocab}\exp(u_w^Tv_c)}{\sum_{w\in Vocab}\exp(u_w^Tv_c)}\\
&= -u_o + \frac{\sum_{w\in Vocab}\exp(u_w^Tv_c)u_w}{\sum_{w\in Vocab}\exp(u_w^Tv_c)} \\
&=-u_o+\sum_{\hat{w}\in Vocab}\frac{\exp(u_\hat{w}^Tv_c)}{\sum_{w\in Vocab}\exp(u_w^Tv_c)}u_\hat{w}\\
&=-Uy+\sum_{\hat{w}\in Vocab}u_\hat{w}\hat{y}_\hat{w}\\
&=-Uy + U\hat{y}\\
&=U(\hat{y}-y)\\\\ &\space where\space U\in\R^{d\times|V|},\space y\in\R^{|V|\times1},\space \hat{y}\in\R^{|V|\times1}
\end{split}
$$
(c) $J_{navie-softmax}(v_c,o,U)=-u_o^Tv_c+\log \sum_{w\in Vocab}\exp(u_w^Tv_c)$
$$
\begin{split}
(w\neq o)\\
\frac{\part J}{\part u_w} &= \frac{\frac{\part}{\part u_w}\sum_{w\in Vocab}\exp(u_w^Tv_c)}{\sum_{w\in Vocab}\exp(u_w^Tv_c)}\\
&= \frac{\exp(u_w^Tv_c)v_c}{\sum_{w\in Vocab}\exp(u_w^Tv_c)}\\
&=\hat{y}_wv_c\\
(w=o)\\
\frac{\part J}{\part u_w} &= -v_c + \frac{\frac{\part}{\part u_w}\sum_{w\in Vocab}\exp(u_w^Tv_c)}{\sum_{w\in Vocab}\exp(u_w^Tv_c)}\\
&= -v_c+\frac{\exp(u_w^Tv_c)v_c}{\sum_{w\in Vocab}\exp(u_w^Tv_c)}\\
&=(\hat{y}_w-1)v_c\\
\end{split}\\
$$

$$
Then \space \space
\frac{\part J(v_c,o,U)}{\part U}=v_c(\hat{y}-y)^T
$$

(d) 
$$
\begin{split}
\frac{\part \sigma(x_i)}{\part x_i} &= \sigma(x_i)-\frac{e^{x_i}\frac{\part}{\part x_i}(e^{x_i}+1)}{(e^{x_i}+1)^2}\\
&= \sigma(x_i) - \frac{e^{2{x_i}}}{(e^{x_i}+1)^2}\\
&= \sigma(x_i) - \sigma^2(x_i)\\
\\
\frac{\sigma(x)}{\part x} &= [\frac{\part\sigma(x_j)}{\part x_i}]_{d\times d}\\
&=
\begin{bmatrix}
\sigma^\prime(x_1)&0&\cdots&0\\
0&\sigma^\prime(x_2)&\cdots&0\\
\vdots&\vdots&\vdots&\vdots\\
0&0&0&\sigma^\prime(x_d)
\end{bmatrix}\\
&=diag(\sigma^\prime(x))
\end{split}
$$
(e) 
$$
J_{neg-sample}(v_c,o,U)=-\log(\sigma(u_o^Tv_c))-\sum_{k=1}^K\log(\sigma(-u_k^Tv_c))
\\
\begin{split}
Respect\space to\space v_c:\\
\frac{\part J}{\part v_c} &= -\frac{\frac{\part}{\part v_c}\sigma(u_o^Tv_c)}{\sigma(u_o^Tv_c)} - \sum_{k=1}^K\frac{\frac{\part}{\part v_c}\sigma(-u_k^Tv_c)}{\sigma(-u_k^Tv_c)}\\
&= -\frac{\sigma(u_o^Tv_c)(1-\sigma(u_o^Tv_c))u_o}{\sigma(u_o^Tv_c)}+\sum_{k=1}^K\frac{\sigma(-u_k^Tv_c)(1-\sigma(-u_k^Tv_c))u_k}{\sigma(-u_k^Tv_c)}\\
&=(\sigma(u_o^Tv_c)-1)u_o+\sum_{k=1}^K\sigma(u_k^Tv_c)u_k\\
\\
Respect\space to\space u_o:\\
\frac{\part J}{\part u_o} &= -\frac{\frac{\part}{\part u_o}\sigma(u_o^Tv_c)}{\sigma(u_o^Tv_c)}\\
&=(\sigma(u_o^Tv_c)-1)v_c\\
\\
Respect\space to\space u_k:\\
\frac{\part J}{\part u_k} &= -\frac{\part}{\part u_k}\sum_{k=1}^K\log(\sigma(-u_k^Tv_c))\\
&=(1-\sigma(-u_k^Tv_c))v_c\\
&=\sigma(u_k^Tv_c)v_c,\space for \space k=1,2,...,K
\end{split}
$$


(f) 
$$
\frac{\part J_{skip-gram}(v_c,w_{t-m},...,w_{t+m},U)}{\part U} = \sum_{-m\leq j\leq m,\space j\neq 0}\frac{\part J(v_c,w_{t+j},U)}{\part U}\\
\frac{\part J_{skip-gram}(v_c,w_{t-m},...,w_{t+m},U)}{\part v_c} = \sum_{-m\leq j\leq m,\space j\neq 0}\frac{\part J(v_c,w_{t+j},U)}{\part v_c}\\
\frac{\part J_{skip-gram}(v_c,w_{t-m},...,w_{t+m},U)}{\part v_w} = 0
$$


### The plot of my training

![word_vectors](/Users/dylan/Downloads/a2/word_vectors.png)

