# CharityML Donor Prediction: Model Selection and Analysis

## Introduction

In this project, a several supervised algorithms will be implemented to accurately model individuals' income using data collected from the 1994 U.S. Census. 
The goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. 

The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

## Model Candidate Overview

### 1. Support Vector Machines (SVM)

- It is used in the classification of diabetes and the possibility of heart attack.
        
-  Well suited for classifiying complex small-meduim sized data.
        
-  Not suitable for large scale data sets (70k+ ) , and it takes too much time to train , sensitive to scalling.
        
-  Since we've got 45k+ samples dataset with some complex relations between features i think its gonna peform well specially with polynomial kernel.

### 2. Ensemble Methods (AdaBoost)

- Used Widely for many machine learning problems and its usually prefered over random forest due to its nature of getting better during training.
        
- AdaBoost gets better by giving more weight to mistakes it made and usually doesn’t get too focused on training data , this is because it uses weighted voting, so no single tree can control the final answer too much.
        
- Because it gives more weight to mistakes, AdaBoost can have trouble with messy or wrong data , also unlike Random Forest which can train many trees at once, AdaBoost must train one tree at a time because each new tree needs to know how the previous trees did. This makes it slower to train.
        
- Since we got few skewed feature histograms , i would prefer it over random forest classifier for this problem , it learns from mistakes as i said , not prone to overfitting as rf and it generalizes better than it in most cases , we've got a meduim sized dataset and not overnoised which makes it suitable for it in my opinion.

### 3. Logistic Regression

- Used widely in very machine learning problems specially with simple tasks that could lead to overfitting if uesd with more complex models.

- It is easy to implement and very efficient to train.

- It requires that each data point be independent of all other data points and it can only predict a categorical outcome.

- Since its a binary classification problem , with balanced number of features , this is the first model that i would think of , simple , very fast to train , and even if it was performing poorly it can give us a quick intution about our problem & performance to decide what do in the next step (as a baseline measurement of performance) .
## Model Selection: Evaluation and Rationale

The selection process focused on identifying the model that best balances predictive performance (particularly F1-score) with computational efficiency for the task of identifying potential donors (> $50k income).

### Evaluation Metrics & Training Time

Evaluation was performed by training models on increasing percentages of the training data and measuring Accuracy and F1-score on a held-out test set. Key observations using 100% of the training data:

*   **AdaBoost Classifier**: Demonstrated the strongest performance, achieving an **F1-score of approximately 75%** and an **accuracy of approximately 85%** on the test set. Training time was efficient.


*   **Logistic Regression**: Achieved an F1-score of approximately 65%. It exhibited the fastest training times, serving as an efficient baseline.


*   **Support Vector Classifier (SVC)**: Also achieved an F1-score of approximately 65%, comparable to Logistic Regression. However, its training time increased dramatically (roughly 10x higher than AdaBoost/LR) when using the full dataset, highlighting its sensitivity to dataset size.

### Algorithm Suitability and Final Selection

*   **Overfitting**: Slight overfitting (difference between training and testing performance) was observed across all models, but AdaBoost maintained the best test set performance.


*   **Trade-offs**: SVC's high computational cost outweighed its performance, which did not surpass AdaBoost. Logistic Regression offered speed but lagged in F1-score. AdaBoost provided the best compromise between high predictive accuracy (especially F1-score, crucial for minimizing wasted outreach efforts) 
and acceptable training/prediction time.


*   **Conclusion**: The **AdaBoost Classifier** was selected as the most appropriate model. Its superior F1-score and accuracy, combined with reasonable computational efficiency on this dataset size, make it the best choice for identifying potential high-income individuals for CharityML.

## Selected Model Performance (AdaBoost)

The selected AdaBoost model underwent hyperparameter tuning to optimize performance.

|     Metric     | Unoptimized Model | Optimized Model |
| :------------: | :---------------: | :-------------: |
| Accuracy Score |      85.76%       |     86.90%      |
| F-score        |      72.46%       |     74.89%      |

*   Optimization yielded improvements: +1.14% in Accuracy and +2.43% in F-score.


*   **Baseline Comparison**: A naive predictor (always predicting the majority class or based on prior probability) achieved an Accuracy of 24.78% and an F-score of 29.17%. The optimized AdaBoost model significantly outperforms this baseline, demonstrating its ability to learn meaningful patterns from the data, which is essential given the F1-score focus for efficient donor identification.

## Feature Importance Analysis (AdaBoost Model)

**Feature Relevance Observation**
1. Age : the age of donator can give a good idea and inution about his per capita income , which can affect the probability of being a candidate for being a donator.


2. Capital Gain : Higher capital gain means higher income of the person , I think it will play a big role, as it is very likely to detect abnormal cases that include wealthy people with high income levels, and therefore they are very likely to donate.


3. Occupation : If you want to get to know a person, the second question you should ask after asking about his name is his job, which gives a quick impression of his financial status in most cases. In my opinion, it is possible to focus on people with high job positions and learn their special patterns to build a model that predicts better.


4. Education : Occupation will be closely linked to the type of education an individual has, and educated people usually make up the largest group of donors.


5. Educaiton Years Number : Years of education will make a difference. A bachelor's degree is not the same as a master's degree. In my opinion, PhD holders will have a very significant impact in supporting donations due to their higher income.



**Feature Relevance Results**

1. Age : I've got it right , the age gives a big intuition of the person is gonna be donor or not , usually older people tend to be donors more than younger people , however age doesn't show that much of interaction when measuring the cumulaitive weight with other features .


2. Hours-per-week : It was actually on my list of recommendations but I eliminated it after carefully considering that the amount of time you invest in work does not necessarily reflect your financial income, simply because most wealthy people do not invest as much time in work as full-time employees. Maybe that wasn't the case here, but in terms of cumulative weighting it would perform better, especially when measured against other factors such as occupation , thus contributing more to the learning process.


3. Capital gain : Capital gains will play a strong role in the nomination of donors, nothing new, but it did not take the largest weight. In fact, I expected it to be in first or second place, perhaps due to the negative impact resulting from capital losses, but its cumulative weight shows that its greatest contribution is with other features, especially since the probability may decrease with the presence of the capital loss indicator on the other hand.


4. Marital Status Married (one-hot encoded) : I never expected that social status would play a role in the learning process, and I would have preferred any other features. The magic of the learning process lies in learning patterns as the model wants them, not as we actually want them to. Logically, it might play a role, and it shows a strong interaction in the cumulative weight with the other features. The effect of social status might be due to the individual’s financial responsibilities towards his family, or some financial freedom gained as a result of being single. Both are ultimately influential factors that will play a strong role in the learning process.


5. Education Num : Years of education play a role, they are mainly related to the age of the individual, more years of education indicates an older age and thus a stronger possibility of being nominated for donation.



*About my misclassified features :* 

In fact, I expected the level of education to have a stronger impact than working hours. The level of education is linked to the occupation, and these two features were not among the top five characteristics for the reasons I explained previously. In fact, it also depends on other factors such as the country and the nature of the work. I believe that the situation would be completely different if this data came from another place (in the Arab countries, for example).

## Effects of Feature Selection


An experiment was conducted by training the optimized AdaBoost model using only the top 5 most important features identified above.

1.  A 4.36 % drop in the A-score performance and a 2% drop in the accuracy score were observed. This is a significant drop, especially given the focus on F-score. There are many patterns that were lost after the feature removal process, especially since the cumulative weight of the top five features is high and reflects their interaction with the rest of the features, regardless of the individual weight of each feature.


2. Actually no, even if time is taken into account, investing time with all the features will lead to better results in the performance of the f1 score. For a charity, the priority is to provide the resources for advertising and attracting donors as much as possible.


*   **Conclusion**: Reducing the feature set to only the top 5 predictors significantly harmed model performance, particularly the F1-score. The patterns captured by the less individually impactful features contribute meaningfully to the overall predictive power, likely through interactions. Therefore, using the **full feature set is recommended** for maximizing the model's effectiveness in identifying potential donors, despite the marginal increase in computation time compared to a reduced set.

## Final Recommendation

The optimized **AdaBoost Classifier**, trained on the **full feature set**, is recommended for CharityML's task. It provides the best balance of high F1-score (critical for efficient donor identification), strong accuracy, and manageable computational performance on the given dataset.
