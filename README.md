# glmnet
Python wrapper around glmnet R package for generalized linear models,
providing scikit-learn style estimators.

The [glmnet](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html) R package was written by the creators of the lasso and elastic net. 
It is highly optimized and tends to outdo other implementations 
in both model performance and speed.  

Currently supports basic linear, logistic, and multinomial (multi-class) models.

To install dependencies:
```
conda install numpy pandas scikit-learn rpy2=2.9.4
Rscript -e 'install.packages("glmnet", repos="https://cloud.r-project.org")'
```
On some systems, it may be necessary to run
```
conda install -c r libiconv
```
prior to the `Rscript` command.

Recent Rpy2 versions have problems with some type conversions.
