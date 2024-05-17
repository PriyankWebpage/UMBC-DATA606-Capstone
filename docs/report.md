
<img width="1000" alt="Main" src="https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/7be5bb19-3c5e-47cb-be02-9c30f8a167c4">


# 1. Title and Author

### Project Title

Wine Quality Prediction

### Prepared for

UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

### Author Name

Priyank Sai Pannem

### Profiles and Presentations

GitHub Profile: [PriyankWebpage](https://github.com/PriyankWebpage)

Linkedin Profile: [Priyank Pannem](https://www.linkedin.com/in/priyankpannem)

# 2. Background

In the field of viticulture and oenology, wine quality is a critical factor in determining market value and customer happiness. Winemakers are continuously looking for ways to improve and preserve the quality of their goods. To this purpose, predictive modeling may be used to forecast wine quality based on physicochemical and sensory characteristics.

### What is it about?
The objective is to build a machine learning model capable of accurately predicting the quality of wine based on its attributes. This model can assist winemakers in assessing the quality of their products and making informed decisions regarding production and refinement processes.

### Why does it matter?

*Quality Assurance:* Predicting wine quality helps ensure consistency and maintain high standards in winemaking. By identifying factors that contribute to good or poor quality, producers can take proactive measures to maintain or improve the overall quality of their wines.

*Consumer Satisfaction:*  Wine consumers have diverse preferences and expectations regarding taste and quality. Predictive models enable producers to tailor their products to meet consumer preferences more effectively, thereby enhancing customer satisfaction and loyalty.

*Cost Optimization:* Predictive modeling can help winemakers optimize production processes and resource allocation. By identifying key factors influencing wine quality, producers can allocate resources more efficiently, reduce waste, and minimize production costs while maintaining or improving quality.

*Market Competitiveness:* In the highly competitive wine industry, maintaining consistent quality is essential for gaining a competitive edge and positioning wines effectively in the market. Predictive models provide valuable insights that can inform marketing strategies and help wineries differentiate their products based on quality.

### What are your research questions?

1. Which physicochemical and sensory attributes have the most significant impact on wine quality?

2. How do different combinations of features contribute to the overall perception of wine quality?

3. How do various machine learning algorithms compare in terms of their ability to predict wine quality accurately?

4. What is the optimal model architecture or ensemble approach for wine quality prediction?

# 3. About Dataset

- Data Sources: [Wine Quality Prediction](https://www.kaggle.com/datasets/subhajournal/wine-quality-data-combined)
- Data size: 2.43 MB
- Data shape:

WineQuality.csv - 32486 rows, 14 columns

### What does each row represent?

The data contains 12 wine features or ingredients based upon which the quality or the types of wine can be predicted. Each row in the dataset represents various chemical properties of wine sample such as chlorides, pH, density, etc.

### 

| Column Name | Data type | Defition | Potential values |
|-----------------|-----------------|-----------------|-----------------|
|Fixed acidity| Numerical | The fixed acidity of a wine, determined by titration, contributes to its overall taste. | |
|Volatile acidity| Numerical | The volatile acidity gives a sour taste to wine and in excess, can lead to an unpleasant vinegar taste. | |
|Citric acid| Numerical |Citric acid is responsible for adding freshness and flavor to wines. | |
|Residual Sugar| Numerical |  Residual sugar refers to the amount of sugar left after fermentation and influences the sweetness of the wine. | |
|Chlorides| Numerical | Chloride concentration can affect the wine's taste and aroma.| |
|Free sulfur dioxide| Numerical |Sulfur dioxide is used as a preservative in winemaking and can influence the wine's aroma and flavor. | |
|Total sulfur dioxide| Numerical | The total amount of sulfur dioxide present in the wine. | |
|Density| Numerical | The density of wine is related to its alcohol content and can indicate its richness. | |
|pH| Numerical | The pH level determines the acidity or basicity of the wine. | |
|Sulphates| Numerical |Sulphates contribute to the wine's antioxidant properties and may affect its taste. | |
|Alcohol| Numerical | The alcohol content is an important factor contributing to the overall balance and taste of the wine.. | |
|Type| Categorical | Type of wine |Red Wine, White Wine |
|Quality| Numerical | The quality of wine is rated on a scale from 1 to 10, with higher values indicating better quality. | |

### Which variable/column will be your target/label in your ML model?
- Quality
### Which variables/columns may be selected as features/predictors for your ML models?
- Fixed acidity, Volatile acidity ,Citric acid, Residual,Sugar, Chlorides, Free sulfur dioxide, Total sulfur dioxide, Density, pH,Sulphates, Alcohol, Type

# 4.Exploratory Data Analysis (EDA)
## 4.1 Data Cleaning

###  4.1.1 Check on Missing values
| Variable              | No. of Null Values |
|-----------------------|-------|
| fixed acidity         | 0     |
| volatile acidity      | 0     |
| citric acid           | 0     |
| residual sugar        | 0     |
| chlorides             | 0     |
| free sulfur dioxide   | 0     |
| total sulfur dioxide  | 0     |
| density               | 0     |
| pH                    | 0     |
| sulphates             | 0     |
| alcohol               | 0     |
| quality               | 0     |
| Type                  | 0     |
- From the above data we can say that there are no signs of null values in the data set
### 4.1.2 Summary Statistics
| Statistics | fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density | pH   | sulphates | alcohol | quality | Type |
|-----------|---------------|------------------|-------------|----------------|-----------|---------------------|----------------------|---------|------|-----------|---------|---------|------|
| count     | 32485.000000  | 32485.000000     | 32485.00000 | 32485.000000   | 32485.000 | 32485.000000        | 32485.000000         | 32485.00| 32485| 32485.000 | 32485.00| 32485.00| 32485|
| mean      | 7.214736      | 0.340122         | 0.318324    | 5.438696       | 0.056009  | 30.458258           | 115.656303           | 0.994719| 3.219| 0.531500  | 10.48069| 5.81169| 0.752|
| std       | 1.308216      | 0.164912         | 0.145152    | 4.799221       | 0.034503  | 17.608076           | 56.456074            | 0.003015| 0.161| 0.148712  | 1.190661| 0.87247| 0.431|
| min       | 3.800000      | 0.080000         | 0.000000    | 0.600000       | 0.009000  | 1.000000            | 6.000000             | 0.987110| 2.720| 0.220000  | 8.000000| 3.00000| 0.000|
| 25%       | 6.400000      | 0.230000         | 0.250000    | 1.800000       | 0.038000  | 17.000000           | 77.000000            | 0.992400| 3.110| 0.430000  | 9.500000| 5.00000| 1.000|
| 50%       | 7.000000      | 0.290000         | 0.310000    | 3.000000       | 0.047000  | 29.000000           | 118.000000           | 0.994900| 3.210| 0.510000  | 10.30000| 6.00000| 1.000|
| 75%       | 7.700000      | 0.410000         | 0.390000    | 8.100000       | 0.065000  | 41.000000           | 156.000000           | 0.997000| 3.320| 0.600000  | 11.30000| 6.00000| 1.000|
| max       | 15.900000     | 1.580000         | 1.660000    | 65.800000      | 0.611000  | 289.000000          | 440.000000           | 1.038980| 4.010| 2.000000  | 14.90000| 9.00000| 1.000|

Summary statistics table of various features related to wine quality.

- **Fixed Acidity:** The average fixed acidity of the wine samples is approximately 7.21, with a standard deviation of 1.31. The minimum and maximum values are 3.8 and 15.9, respectively.

- **Volatile Acidity:** The average volatile acidity is around 0.34, with a standard deviation of 0.16. The values range from 0.08 to 1.58.

- **Citric Acid:** The mean citric acid content is approximately 0.32, with a standard deviation of 0.15. The minimum value is 0, and the maximum value is 1.66.

- **Residual Sugar:** The average residual sugar content is about 5.44, with a standard deviation of 4.80. The values range from 0.6 to 65.8.

- **Chlorides:** The mean chloride concentration is approximately 0.056, with a standard deviation of 0.035. The minimum and maximum values are 0.009 and 0.611, respectively.

- **Free Sulfur Dioxide:** The average free sulfur dioxide content is around 30.46, with a standard deviation of 17.61. Values range from 1 to 289.

- **Total Sulfur Dioxide:** The mean total sulfur dioxide content is about 115.66, with a standard deviation of 56.46. The values range from 6 to 440.

- **Density:** The average density of the wine samples is approximately 0.995, with a standard deviation of 0.003. The density ranges from 0.987 to 1.039.

- **pH:** The mean pH level is around 3.22, with a standard deviation of 0.16. The values range from 2.72 to 4.01.

- **Sulphates:** The average sulphate concentration is about 0.53, with a standard deviation of 0.15. The minimum and maximum values are 0.22 and 2.0, respectively.

- **Alcohol:** The mean alcohol content is approximately 10.48%, with a standard deviation of 1.19. Values range from 8 to 14.9.

- **Quality:** The quality score of the wine samples has a mean of around 5.81, with a standard deviation of 0.87. The values range from 3 to 9.

- **Type:** This column indicates the type of wine, with 1(White wine) representing one type and 0(Red wine) representing another type. The majority of the samples seem to belong to type 1, as indicated by the mean value being close to 1.

  ## 4.2 Data Preprocessing
  
  For a number of reasons, standardizing the data in the wine dataset is essential. First of all, it guards against biases favoring variables with wider ranges or units by guaranteeing that variables with varying scales and units contribute equally to the study. Second, standardization makes it simpler to compare and understand coefficients in statistical models, which in turn makes determining the relative significance of various characteristics easier. Additionally, by lessening the influence of outliers and enhancing convergence, standardizing the data can enhance the effectiveness of machine learning systems. Standardized coefficients, which indicate the change in the response variable per standard deviation change in the predictor, further improve the interpretability of the results. Overall, standardizing the data in the wine dataset enhances the reliability, interpretability, and generalizability of analyses and model, thus preventing data skewness that might be caused by the range of values.

  ## 4.3 Plots

     ### 4.3.1 Correalation Heat Map
      
  ![corr](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/cf3f8ee9-ecb5-4517-aa74-6f7110665770)

    **Summary Insights**
    - From the above correlation matrix it is observed no two independent variables are highly corealted with eachother as all the co realtional values are less than 0.75
    - This shows there is no sign of multicollinearity, with this we can move forward without eliminating any columns.
      
 
      ### 4.3.2 Density Plots
      
      ![image](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/209047fd-c9c2-4f1c-baf7-83a0a70df4e2)
     **Summary Insights**
      - The plot is a series of density plots for various features of a dataset, likely related to wine characteristics given the labels. Each subplot shows the distribution of a specific variable. Here's a detailed explanation of each subplot:

        **Fixed Acidity:**
        - Most wines have a fixed acidity between 6 and 8, with a prominent peak around 7. This suggests a common range for this characteristic in the dataset.
        
        **Volatile Acidity:**
        - The majority of wines have volatile acidity around 0.3, indicating that high volatile acidity levels are less common. Lower volatile acidity is likely preferred as higher levels can negatively impact taste.
    
        **Citric Acid:**
      
        - The bimodal distribution suggests there are two common levels of citric acid in wines: one near 0 and another around 0.5. This may reflect different styles or types of wine.
    
        **Residual Sugar:**
        
        - The skewness indicates that most wines have low residual sugar levels, but there are a few with significantly higher levels, potentially indicating sweet wines or dessert wines.
        
        **Chlorides:**
        
        - Most wines have low chloride levels, centered around 0.05. High chloride levels are rare, which is positive since high chloride can indicate poor quality.
        
        **Free Sulfur Dioxide:**
        
        - There is a concentration around 30 mg/L, with fewer wines having higher levels. Proper sulfur dioxide levels are             crucial for preventing spoilage and oxidation.
        
        **Total Sulfur Dioxide:**
        
        - A peak around 150 mg/L shows where most wines fall, but a wide distribution suggests varied winemaking practices regarding sulfur dioxide usage.
        
        **Density:**
        
        - Density values are tightly clustered around 0.99 to 1.0, indicating consistency in the wine’s composition, primarily due to the alcohol and sugar content.
        
        **pH:**
        
        - The nearly normal distribution around 3.2 indicates most wines have a balanced pH, which is important for stability and taste.
        
        **Sulphates:**
        
        - Most wines have sulphate levels around 0.5, which can contribute to the antimicrobial and antioxidant properties in wine.
        
        **Alcohol:**
        
        - The distribution shows several peaks, with the most common alcohol content around 9.5%, but a wide range up to 15%. This reflects the diversity in wine styles and strength.
        
        
        **Quality:**
        
        - Quality ratings show distinct peaks at 5, 6, and 7, suggesting that these are the most common quality ratings in the dataset. This indicates that the majority of wines are rated around average to slightly above average quality.
        
        Overall, these plots provide a comprehensive overview of the distribution of each feature in the dataset, highlighting the central tendencies, variability, and presence of outliers.

### 4.3.3 Pair Plots

  ![image](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/b6db0cb2-2a09-4cff-910c-5fa323f03144)
  
  **Summary Insights**
  
  - 1(Orange) indicates white wine

  - 0(Blue) indicates red wine

### 4.3.4 Box Plots
  ![image](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/e2fe603d-f9d8-42e0-b2fd-61f4794f0b05)

  **Summary Insights**

- Type 1 represents **White Wine**

- Type 0 represents **Red Wine**

- White Wine generally exhibits higher values for volatile acidity, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, and alcohol content compared to Red Wine.

- Red Wine tends to have higher fixed acidity and citric acid content.

- Features like density and quality do not show significant differences between the two types, indicating they might not be strong differentiators.

- White Wine often shows greater variability in many features, suggesting a broader range of values within this type.

### 4.3.5 Distribution of Wine Quality

![dist](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/151d54ec-55b3-4439-affa-98530c3d9437)

  **Summary Insights**

- The output shows the distribution of wine quality ratings in the dataset. Each quality rating (ranging from 3 to 9) has a corresponding count of how many instances in the dataset have that rating.

- In this case, it's evident that the dataset is imbalanced, meaning that some quality ratings are much more prevalent than others. For example:

- Quality ratings of 5 and 6 are the most common, with 10,755 and 14,117 instances, respectively.

- Quality ratings of 3, 4, 8, and 9 are relatively rare, with counts ranging from 24 to 1,144 instances.

- Imbalanced data can pose challenges for machine learning models, especially for classification tasks, as they might become biased towards the majority class.

### 4.3.6 Percentage of Best Quality Categories
  
  ![pichart](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/f7bbd46f-63f2-4b47-bbce-7f579e352bb4)
 **Summary Insights**
  
- "Good quality": 63%

- "Poor quality": 37%

- While it's clear that the dataset is not perfectly balanced (i.e., the proportions are not exactly 50/50).

- In many cases, a slight imbalance like this may not significantly impact the analysis, particularly if the class imbalance is not severe and the chosen machine learning algorithms can handle it effectively.

### 4.3.7 Stacked Bar Plot of Quality by Type

![newplot](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/fd89c85f-8b6c-43b6-ab6e-b65c7144823b)

**Summary of Insights:**

- White wines dominate the dataset and are most commonly rated at 5 and 6, but they also have a notable presence at the lower quality ratings.

- Red wines are fewer in number but tend to score higher quality ratings, especially at the top end (7 and 8).

- Quality ratings of 5, 6, and 7 cover the majority of the wines, with White wines being more common in this range, particularly at 5 and 6.

### 4.3.8 Mean and Median Alcohol Content by Quality

![newplot1](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/ea08cdbf-2cfd-4b8a-99ae-e4d00924cef6)

**Summary of Insights:**

- There is a clear trend showing that higher quality wines have higher alcohol content.

- The close alignment of mean and median values across most quality ratings suggests a balanced distribution of alcohol content within each category.

- The increasing trend in alcohol content from lower to higher quality ratings indicates that alcohol content could be a significant factor in determining the quality of wine.

- These insights can help in understanding the relationship between alcohol content and perceived quality of wine, suggesting that wines with higher alcohol content tend to be rated higher in quality.



        
# 5. Model Training

### 5.1. Models used for Predictive Analytics

  - Logistic Regression
  - XGboost
  - Random Forest

### 5.2 Packages Used
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Plotly
  - Scikit-learn
  - XGBoost

### 5.3 Development Environments
  - VS Code
  - Anaconda
  - GitHub

### 5.4 Model workflow
  - The wine quality prediction model follows a systematic machine learning workflow to ensure robust and accurate predictions. Here is a summary of the key steps involved:

**Data Splitting:**

- The dataset is split into training and testing sets with a 70:30 ratio, ensuring a sufficient amount of data for both training the model and evaluating its performance.

**Pipeline Creation:**

- A machine learning pipeline is constructed using StandardScaler to standardize the features. This scaling ensures that all features contribute equally to the model and improves the model's performance.
Various machine learning algorithms are tested within this pipeline to identify the best performing model.

**Model Selection:**

  **5.4.1 Logistic Regression**
  
   - Logistic regression can be effectively used for wine quality prediction by modeling the relationship between various features (such as acidity, sugar content, pH, etc.) and the binary target variable indicating wine quality (e.g., good or poor quality). As a binary classifier, logistic regression estimates the probability that a given wine sample belongs to a particular quality class. The model applies the logistic function to a linear combination of input features, producing an output between 0 and 1, which can be interpreted as the probability of the wine being of good quality.

  **Classification Report on Training Data**

|   Class  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Poor Quality Wine    | 0.69      | 0.56   | 0.62     | 8422    |
| Good Quality Wine    | 0.77      | 0.85   | 0.81     | 14317   |
| Accuracy  |           |        | 0.74     | 22739   |
| Macro Avg | 0.73      | 0.70   | 0.71     | 22739   |
| Weighted Avg | 0.74   | 0.74   | 0.73     | 22739   |



**Classification Report on Testing Data**

|      Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Poor Quality Wine    | 0.69      | 0.56   | 0.62     | 3610    |
| Good Quality Wine    | 0.77      | 0.85   | 0.81     | 6136    |
| Accuracy  |           |        | 0.75     | 9746    |
| Macro Avg | 0.73      | 0.71   | 0.72     | 9746    |
| Weighted Avg | 0.74   | 0.75   | 0.74     | 9746    |


**Confusion Matrix of Logestic Regression**

  ![image](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/0cf4e42a-b069-4815-ac89-9aca3d637aa8)
  
  - **True Positives (TP):** 5237
  
      These are instances where the model correctly predicted the positive class. In this case, it correctly identified 5198 instances as "positive" or "good quality" wine.
  
  - **False Positives (FP):** 1572
  
      These are instances where the model incorrectly predicted the positive class when the actual class was negative. In other words, the model predicted 1567 instances as "positive" or "good quality" wine when they were actually "negative" or "poor quality" wine.
  
  - **True Negatives (TN):** 2038
  
      These are instances where the model correctly predicted the negative class. It correctly identified 2043 instances as "negative" or "poor quality" wine.
  
  - **False Negatives (FN):** 899
  
      These are instances where the model incorrectly predicted the negative class when the actual class was positive. In other words, the model predicted 938 instances as "negative" or "poor quality" wine when they were actually "positive" or "good quality" wine.
  
      In summary, while the model demonstrates some effectiveness in identifying instances of "positive" or "good quality"         wine, there is room for improvement, particularly in reducing false positives and false negatives.


**Area under the Curve**

![image](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/d9cd05ce-b104-443e-8fd5-d211dd8ab30e)

- An AUC score of 0.81 suggests that the model has some predictive capability, but further optimization or adjustments may be necessary to enhance its performance, especially aming at higher accuracy.

**5.4.2 XGBClassifier**

XGBoost Classifier (XGBClassifier) can estimate wine quality using its sophisticated gradient boosting method. Initially, the dataset goes through preparation stages including managing missing values and encoding categorical variables. The XGBClassifier model is then trained using features such as acidity, sugar content, and pH. During training, XGBoost iteratively constructs numerous decision trees, each fixing the faults of the preceding one. It improves the model's performance by minimizing a predetermined loss function, therefore efficiently capturing complicated patterns in the data. The model's hyperparameters, such as the learning rate and maximum tree depth, may be fine-tuned using approaches such as GridSearchCV to improve forecast accuracy. Finally, the trained XGBClassifier can reliably determine the quality of wines based on their chemical makeup.

**Classification Report on Training Data of XGBClassifier**

|  Class  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Poor Quality Wine   | 0.92      | 0.90   | 0.91     | 8422    |
| Good Quality Wine    | 0.94      | 0.95   | 0.95     | 14317   |
| Accuracy  |           |        | 0.93     | 22739   |
| Macro Avg | 0.93      | 0.93   | 0.93     | 22739   |
| Weighted Avg | 0.93   | 0.93   | 0.93     | 22739   |

**Classification Report on Testing Data of XGBClassifier**

| Class | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Poor Quality Wine   | 0.90      | 0.88   | 0.89     | 3610    |
| Good Quality Wine   | 0.93      | 0.94   | 0.94     | 6136    |
| Accuracy  |           |        | 0.92     | 9746    |
| Macro Avg | 0.91      | 0.91   | 0.91     | 9746    |
| Weighted Avg | 0.92   | 0.92   | 0.92     | 9746    |

**Confusion Matrix of XGBClassifier**

![image](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/6aec4ee9-6e9b-4810-92db-1c0382830a9d)

- **True Positives (TP):** 5723

    The model correctly predicted 5723 instances of "Good quality" wine. These are cases where the wine was actually of the good quality, and the model correctly identified it as such.

- **False Positives (FP):** 446

    The model incorrectly predicted 446 instances of "Poor quality" wine as "Good quality" wine. These are cases where the wine was not actually of the good quality, but the model mistakenly classified it as such.

- **True Negatives (TN):** 3164

    The model correctly predicted 3164 instances of "Poor quality" wine. These are cases where the wine was not of the good quality, and the model correctly identified it as such.

- **False Negatives (FN):** 413

    The model incorrectly predicted 413 instances of "Good quality" wine as "Poor quality" wine. These are cases where the wine was actually of the good quality, but the model failed to identify it correctly.

    - Interpreting these values in the context of wine quality, we can see that the model has a relatively high number of true positives, indicating that it is effective at identifying wines of the good quality. However, it also has a noticeable number of false positives and false negatives, suggesting that there are areas for improvement in its ability to distinguish between wines of different qualities. Further analysis, model refinement, and possibly the incorporation of additional features could help improve the model's performance in predicting wine quality accurately.

**Area under the Curve**

  ![image](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/ae76daba-b26e-4277-ae39-b8be67acc943)

  - AUC score of 0.9713 implies that the model has a high true positive rate while maintaining a low false positive rate across different threshold settings. This indicates that the model is performing exceptionally well in correctly classifying instances of both classes and is making very few mistakes in misclassifying instances.

  **5.4.3 Random Forest Classifier**
  - The Random Forest Classifier uses ensemble learning to estimate wine quality. Initially, the dataset is preprocessed to accommodate missing values and encode categorical characteristics. The Random Forest model is then trained using characteristics such as acidity, sugar concentration, and pH. During training, the classifier generates numerous decision trees from random subsets of the data and features, guaranteeing variety and decreasing overfitting. Each tree separately forecasts wine quality, and the ultimate prediction is established by pooling all trees' votes. The model's hyperparameters, such as the number of trees and maximum depth, may be tuned using GridSearchCV. Overall, the Random Forest Classifier provides robust and reliable predictions about wine quality based on its chemical makeup.

**Classification Report on Training Data**

|  Class  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Poor Quality Wine  | 1.00      | 1.00   | 1.00     | 8422    |
| Good Quality Wine   | 1.00      | 1.00   | 1.00     | 14317   |
| Accuracy  |           |        | 1.00     | 22739   |
| Macro Avg | 1.00      | 1.00   | 1.00     | 22739   |
| Weighted Avg | 1.00   | 1.00   | 1.00     | 22739   |


   **Classification Report on Testing Data Random Forest Classifier**

|  Class   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Poor Quality Wine   | 1.00      | 0.99   | 0.99     | 3610    |
| Good Quality Wine    | 1.00      | 1.00   | 1.00     | 6136    |
| Accuracy  |           |        | 1.00     | 9746    |
| Macro Avg | 1.00      | 0.99   | 1.00     | 9746    |
| Weighted Avg | 1.00   | 1.00   | 1.00     | 9746    |

**Confusion Matrix of Random Forest Classifier**
![image](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/a9fff0c7-15b5-4ac5-a5ad-f6ce5a60b0a7)

  - **True Positives (TP):** 6119
  
      The model correctly predicted 6119 instances of "good quality" wine. These are cases where the wine was actually of good quality, and the model correctly identified it as such.
  
  - **False Positives (FP):** 17
  
      The model incorrectly predicted 17 instances of "poor quality" wine as "good quality" wine. These are cases where the wine was not actually of good quality, but the model mistakenly classified it as such.
  
  - **True Negatives (TN):** 3593
      
      The model correctly predicted 3593 instances of "poor quality" wine. These are cases where the wine was not of good quality, and the model correctly identified it as such.
  
  - **False Negatives (FN):** 17
      
      The model incorrectly predicted 17 instances of "good quality" wine as "poor quality" wine. These are cases where the wine was actually of good quality, but the model failed to identify it correctly.
  
    This confusion matrix still demonstrates a highly accurate model with very few misclassifications. It effectively identifies both "good quality" and "poor quality" wines, with only a small number of false positives and false negatives. Overall, the model appears to be reliable for classifying wine quality.

**Area under the Curve**

![image](https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/6e5ea8f0-b689-47d8-be3b-8873dc84c483)

  - Given the high AUC score of 1.0, it suggests that the model has strong predictive power and is capable of making accurate classifications. This indicates that the model is likely well-calibrated and provides reliable predictions for the given task of classifying wine quality.
  
  - Among all other models Random forest model has showed better performance with 100% F1 score over the Logistic Regression and XGBoost

- In summary, Random Forest achieved perfect performance, XGBoost performed slightly less accurately but still well, while Logistic Regression showed the lowest accuracy among the three classifiers. Depending on the F1 score , one might choose the Random Forest as best performing model.
  

**Hyperparameter Tuning:**

 - GridSearchCV with a 5-fold cross-validation (KFold) is used to find the optimal hyperparameters for the RandomForest classifier. This method helps in ensuring that the model generalizes well to unseen data and avoids overfitting.

**Model Evaluation:**

 - The final model's performance is evaluated using the f1 score, a metric that balances precision and recall, providing a comprehensive measure of the model's predictive power.

This structured approach ensures that the wine quality prediction model is robust, well-tuned, and capable of making accurate predictions on new data.

## 6. Web Application 

- Created a web application on Streamlit to monitor and improve the quality of wine production can provide numerous benefits across different stages of the wine-making process.
- Streamlit is an excellent tool for creating interactive web applications for machine learning models. For wine quality prediction, I developed a Streamlit app that allows users to input various chemical properties of the wine (such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol). The app then uses a user trained machine learning model to predict the quality of the wine.
- To build the app,I installed Streamlit. Saved my trained model (e.g., a RandomForestClassifier) as pickle file. Later created a Streamlit app (app.py) that loads this model and provides an input form for the user to enter the wine's properties. When the user submits the form, the app will display the predicted wine quality. This approach offers an accessible interface for users to leverage machine learning models for wine quality prediction without needing extensive technical knowledge.
- The below image gives the idea how user can interact with the web application. After user enters the values of chemical features of the wine sample and clicks on predict then this data is sent to the model where the model will predict wheather the given wine sample is good or bad quality.
  <img width="998" alt="Web app" src="https://github.com/PriyankWebpage/UMBC-DATA606-Capstone/assets/65448205/d1f856c5-de5e-4944-b551-1939a0cb6ea5">

# 7. Conclusion

- The goal of this research is to create a machine learning model that can predict wine quality based on its chemical attributes. The workflow consists of data pretreatment, exploratory data analysis (EDA), feature engineering, model selection, hyperparameter tweaking, and evaluation. A Random Forest classifier is effective in predicting wine quality and differtiating between poor and good quality wines, which yielded good performance. The model was then implemented using Streamlit to construct a web application that allows users to enter chemical characteristics from a wine sample and obtain a quality forecast. As we advance, refining these models with more complex datasets and newer technologies presents an exciting opportunity. Let's embrace the future of technology in crafting the perfect wine.

  ### 7.1 Potential Application
   - The primary application is in the wine industry, where it can be utilized by vintners to promptly evaluate the quality of wine batches depending on chemical composition, thereby guaranteeing consistent product quality. It can also prove beneficial for quality control laboratories seeking to automate and streamline testing procedures, potentially resulting in cost reductions and efficiency enhancements. Automating certain steps through machine learning modeling may help to objectively and reproducibly assess wine quality while eliminating subjective human factors from consideration. This could help to standardize testing and allow for easy comparison of analysis results over time. Ultimately, consistently delivering high quality wines remains a top priority for any winery. The methods explored here offer wine producers a valuable tool in maintaining the highest product standards for their customers.

  ### 7.2 Limitations
  - **Data Imbalance** :The dataset utilized for model training have exhibited data imbalance, where the distribution of data points across the target variable's classes was uneven. This could potentially skew the model's predictions towards the majority class, diminishing its ability to accurately predict the minority class. Further analysis is needed to assess whether the dataset exhibited class imbalance and, if so, whether appropriate techniques such as oversampling or undersampling were applied to address it.

  - **Feature Importance**: It is possible that not all of the features, or variables, included in the dataset provided equal value in predicting the target variable of wine quality. The model building process may have attributed undue weight to certain features that were less relevant or informative, while overlooking features of greater importance. A thorough analysis of each feature's predictive power could help in determining if the model relied too heavily on less impactful inputs.

  - **Overfitting**: Even after implementing regularization and validation techniques intended to prevent overfitting, there remains a risk that the trained model may not generalize conclusively to new, unseen data due to overfitting its predictions to the idiosyncrasies of the original training data. Further testing on hold-out evaluation datasets is needed to evaluate the model's true ability to generalize.

  - **Limited Features**: The model developed could only take into account the features, or variables, that were included in the original dataset. Other factors known to influence wine quality, such as grape varietal, vinification processes, and terroir characteristics, were not available to the model. Incorporating additional relevant data could potentially result in a more holistic and higher performing model.
    
  ### 7.3 Lessons learned
  
  - **Data Preprocessing:** The importance of thorough data cleaning and preprocessing cannot be overstated, as it significantly impacts model performance. Prior to model training, data should be examined for missing values, outliers, inconsistencies or other anomalies. Appropriate techniques such as normalization, feature engineering and feature selection should then be applied to ensure the data is clean, consistent and suitable for accurate modeling. Thorough preprocessing helps reduce noise in the data and results in more robust and generalizable models.

  - **Model Evaluation:** Evaluating models using multiple metrics is crucial to obtain a comprehensive understanding of their performance. Common metrics like accuracy, precision, recall and F1 score provide insight into a model's predictions. However, to assess overall effectiveness, it is best practice to examine multiple complementary metrics. Additional analyzes such as confusion matrices, ROC curves can also uncover important information about how a model is performing that single-value metrics may not reveal. Comprehensive evaluation allows practitioners to determine the most suitable model for a given problem or use case.

  - **Hyperparameter Tuning:** Effective hyperparameter tuning can significantly improve model performance, though it can be computationally expensive. Hyperparameters control aspects of model training like complexity, regularization and optimization technique. Their values should be carefully selected to avoid under or overfitting. Grid search, random search and Bayesian optimization are common techniques for automated tuning. While computationally intensive, taking the time to tune hyperparameters using validation data often results in enhanced generalization and real-world effectiveness. 

  ### 7.4 Future Research Direction
  - **Feature Expansion for Improved Model Accuracy:** The inclusion of additional features related to wine quality has the potential to improve the accuracy of our predictive models. Capturing information about factors like grape variety, region of origin, and aging process may provide useful context when assessing characteristics such as taste and price. By expanding the feature set to incorporate more qualitative elements involved in wine production, we aim to develop models with a more nuanced understanding of this complex domain.

  - **Evaluation of Advanced Modeling Techniques:** While our current models have shown promising results, exploring more sophisticated approaches may yield even better predictive performance. Techniques like neural networks and ensemble methods have demonstrated strength on related problems and warrant further investigation. These advanced algorithms may be able to capture non-linear relationships and interdependencies in a way that could push the boundaries of what is achievable with our existing models.

  - **Implementing MLOps for Streamlined Model Deployment and Maintenance:** To fully capitalize on our modeling efforts, it is important to implement MLOps practices that facilitate reliable, efficient model operations. This includes automating end-to-end data and model workflows, setting up continuous integration and deployment pipelines, and putting in place monitoring systems. Adopting an MLOps mindset will help ensure model outputs can be consistently reproduced and integrated into our workflows as we expand our capabilities to new platforms and problem domains over time.


# References
-  Scikit-learn Documentation: https://scikit-learn.org/stable/
-  Streamlit Documentation: https://docs.streamlit.io/
-  XGBoost Documentation: https://xgboost.readthedocs.io/en/stable/
-  Wine Quality Dataset: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
-  GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
