# glmnet
Python wrapper around glmnet R package for generalized linear models,
providing scikit-learn style estimators.

The [glmnet](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html) R package was written by the creators of the lasso and elastic net. 
It is highly optimized and tends to outdo other implementations 
in both model performance and speed.  

Currently supports basic linear, logistic, and multinomial (multi-class) models.

To install dependencies:
```
conda install numpy pandas scikit-learn rpy2
Rscript -e 'install.packages("glmnet", repos="https://cloud.r-project.org")'
```
