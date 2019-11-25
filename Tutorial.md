## 1.	Logistic Regression:

In statistics, the __logistic model__ (or logit model) is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. 
This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc... Each object being detected in the image would be assigned a probability between 0 and 1 and the sum adding to one. 

In our case, the __daily return of SP500__ is classified as

 __“2”: daily return >0.1%__
 
 __“1”: -0.1%<= daily return <= 0.1%__
 
 __“0”: daily return < -0.1%__

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model. 

Thus, we regress on __daily return__ with respect to the __score of 8-K documents__, to see if the high score of 8-k file would lead to high performance of the stock. 

* __Why not use regular linear regression?__

We cannot fit the regular linear regression because the line fitted will be above 2 and below 0, but that does not make sense.

## 2.	Random Forest

The random forest combines hundreds or thousands of decision trees, trains each one on a slightly different set of the observations, splitting nodes in each tree considering a limited number of the features. The final predictions of the random forest are made by averaging the predictions of each individual tree.

Random forests use a modified tree learning algorithm that selects, at each candidate split in the learning process, a random subset of the features. This process is sometimes called "feature bagging". The reason for doing this is the correlation of the trees in an ordinary bootstrap sample: if one or a few features are very strong predictors for the response variable (target output), these features will be selected in many of the B trees, causing them to become correlated. 

In our case, we optimize the random forest using the __RandomizedSearchCV__ in Scikit-Learn. Optimization refers to finding the best hyperparameters for a model on a given dataset. 
Examples of what we optimize in a random forest are the number of decision trees, the maximum depth of each decision tree, the maximum number of features considered for splitting each node, and the maximum number of data points required in a leaf node.

* __Why use cross-validation for random forest?__

To resolve the __overfitting__ issue: Even though adding more features to our regression would help make more precisely fitted line, however, the line may only fit to the existing data point instead of truly capture the pattern of the data. Therefore, by using cross-validation on training data, we can tell if the fitted line truly predicts the future data. If the MSE of training data and out-of-sample data is similar, we can conclude that there is no overfitting-issue. However, if the MSE of out-of-sample data is larger than training data MSE, there is an overfitting issue.


