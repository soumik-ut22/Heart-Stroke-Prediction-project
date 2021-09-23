# Heart-Stroke-Prediction-project

# Problem Statement
What are the important factors in predicting strokes and how can hospitals/public health use this information to prevent strokes and help identify high risk people and let them know what to do if they experience a stroke.
https://www.stroke.org/en/about-stroke

# Initial findings and thoughts behind it
There was a total of 5000 total observations. Since it was a small dataset, we had issues with skewed classes and this impacted how we ran our model.
We decided to oversample the under-represented class to train and validate.
We ran boosting with all variables - got important variables and understood the issue with the class size (poor performance in validation).
We prioritized recall over precision to lessen likelihood of false negatives (which would potentially risk the lives of people). 

# Predictive models tried: 
KNN - Qualitative, was a no (b/c most predictors were categorical. And could not calc. distances between cat. params)
Random Forest 
Bagging
Boosting & Threshold 

# Model achievements
The most important predictors for strokes are age, body mass index, and average glucose level.
Our best model can predict 28% of strokes, with a 93.2% accuracy based off the data made available. 

# Conclusions
Hospitals can use the model to alert patients if they are likely to experience a stroke and what they can to to lower their chances
Yearly physicals are a perfect opportunity to use the model
Hospitals or public health agencies can create pamphlets explaining what to do if you are experiencing a stroke and place them in areas where the people likely to have strokes according to the model would be.

