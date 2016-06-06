This is the solution to the Quora data challenge as found in https://www.quora.com/challenges#answer_classifier.

This is a blending of 7 classifier with the majority voting scheme. This project made use of cross validation to prevent overfit. The best parameters are usually obtained using hyper parameter search.

The result was greatly improved by feature selection. The preprocessing step involves using the feature selection for model that that leads to improvement and using whole feature where it leads to reduction in quality.

Alway trust your CV score.

The result of prediction can be new-output.txt
