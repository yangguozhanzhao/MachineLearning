#Machine Learning Study
> coursera Andrew Ng [machine-learning](https://www.coursera.org/learn/machine-learning/home/welcome)  

## Linear Regression
* Normal Equations
	$$\theta = (X^TX)^{-1}X^T\vec{y}$$
* Gradient Descent  
    - minimize the cost function  
    $$J(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y{(i)})^2$$
    $$h_\theta(x) = \theta^Tx=\theta_0+\theta_1x_1$$ 
    - iteration
    $$ \theta_j = \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$$

##Logistic Regression
* Cost function
$$ J(\theta) = \frac{1}{m}\sum_{i=1}^m[-y^{(i)}\log(h_\theta(x^{(i)}))-(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$
$$ h_\theta(x) = g(\theta^Tx) \qquad g(z)=\frac{1}{1+e^{-z}}$$
* Gradient
$$\frac{\partial J(\theta)}{\partial \theta_j}=\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)}))x_j^{(i)} $$
> fminunc:instead of taking gradient descent steps, use an Octave/-
MATLAB built-in function called fminunc.
* Regularized logistic regression ($\theta_0=0$)
$$ J(\theta) = \frac{1}{m}\sum_{i=1}^m[-y^{(i)}\log(h_\theta(x^{(i)}))-(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2 \quad for \quad j\geq 1$$
$$\frac{\partial J(\theta)}{\partial \theta_j}=\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)}))x_j^{(i)}+\frac{\lambda}{m}\theta_j \quad for \quad j\geq 1$$
>larger $\lambda$,Underfitting  
>smaller $\lambda$,Overfitting  

## Neural Networks
* Cost function
$$ J(\theta) = \frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K[-y_k^{(i)}\log(h_\theta(x^{(i)})_k)-(1-y_k^{(i)})\log(1-h_\theta(x^{(i)})_k)]$$

* Regularized cost function
$$ J(\theta) = \frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K[-y_k^{(i)}\log(h_\theta(x^{(i)})_k)-(1-y_k^{(i)})\log(1-h_\theta(x^{(i)})_k)] +\frac{\lambda}{2m}[\sum_{j=1}^{25}\sum_{k=1}^{400}(\Theta^{(1)}_{j,k})^2+\sum_{j=1}^{10}\sum_{k=1}^{25}(\Theta^{(2)}_{j,k})^2]$$

* Backpropagation for gradient
    * Random initialization
    * Gradient checking




