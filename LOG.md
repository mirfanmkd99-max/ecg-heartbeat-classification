=> Load data

=> Split into features and labels

=> Analyze shape 

=> Checked for missing values

=> There was a class imbalance 80% were from normal class

=> The beats were varying across classes

&#x20;



=> After that I split training and validation data of 15% from the original data





=> Train Random Forest Classifier with balance class weights and without class weights. There was a slight difference in the Macro F1 score. 



=> For experiments, I did train different models like Histogram Gradient Boosting Classifier, Logistic Regression and then using Oversampling up to 20k samples each class.



=> Boosting and oversampled Random Forest Classifier were the efficient ones in weights of F1.







