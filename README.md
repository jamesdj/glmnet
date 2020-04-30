# glmnet
Python wrapper around glmnet R package for generalized linear models,
providing scikit-learn style estimators.

Supports linear, logistic, and multinomial (multi-class) models.

To install dependencies:
```
conda install numpy pandas scikit-learn rpy2
Rscript -e 'install.packages("glmnet", repos="https://cloud.r-project.org")'
```
