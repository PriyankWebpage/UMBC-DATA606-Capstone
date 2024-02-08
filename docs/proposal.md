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
Data shape:

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
|Type| Numerical | The fixed acidity of a wine, determined by titration, contributes to its overall taste. | |
|Quality| Numerical | The fixed acidity of a wine, determined by titration, contributes to its overall taste. | |




