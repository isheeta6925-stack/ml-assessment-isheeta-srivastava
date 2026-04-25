B1. PROBLEM FORMULATION
(a) Machine Learning Problem
The objective is to predict the number of items sold for each store under different promotional strategies. The target variable is items_sold, while the input features include store characteristics (store size, location type, competition density), promotion type, and temporal features such as month and day of week.
This is a supervised machine learning regression problem because the goal is to predict a continuous numerical value based on historical data. Regression is appropriate since items_sold is a quantitative variable.

(b) Why items_sold instead of revenue
Using items_sold as the target variable provides a more stable and reliable measure of promotion effectiveness compared to total revenue. Revenue can be influenced by pricing variations, discounts, and product mix, which may distort the true impact of a promotion. In contrast, items_sold directly reflects customer response in terms of volume.
This illustrates the broader principle that the target variable in machine learning should closely align with the business objective while minimizing noise and external distortions.

(c) Better modelling strategy
Instead of using a single global model, a more effective approach would be to use either a segmented modelling strategy or a hierarchical model. For example, separate models can be trained for urban, semi-urban, and rural stores, or store-specific features can be incorporated more strongly.
This approach accounts for regional differences in customer behavior and ensures that the model captures local variations in response to promotions.

B2. DATA & EDA STRATEGY
(a) Data joining
The data from transactions, store attributes, promotion details, and calendar tables would be joined using common keys such as store_id, promotion_type, and transaction_date.
The final dataset would have a grain of one row per store per day (or per transaction period). Aggregations such as total items sold, average basket size, and total visits per period may be computed before modelling.
This ensures that each row represents a meaningful unit for prediction and avoids duplication or inconsistency.

(b) EDA strategy
Several exploratory analyses would be conducted before modelling:
1. Distribution of items_sold to understand skewness and outliers.
2. Boxplots of items_sold across different promotion types to evaluate their effectiveness.
3. Correlation heatmap to identify relationships between numerical variables such as competition density and sales.
4. Time-series plots of sales to detect seasonality and trends.
These analyses help in identifying important features, detecting anomalies, and guiding feature engineering decisions such as creating temporal variables or transforming skewed features.

(c) Imbalance issue
If 80% of transactions occur without promotions, the dataset is imbalanced. This may cause the model to under-learn the impact of promotional strategies.
To address this, techniques such as resampling, assigning higher weights to promotional observations, or ensuring balanced representation during training can be used. Additionally, evaluation metrics should focus on performance across different promotion types.

B3. MODEL EVALUATION & DEPLOYMENT
(a) Train-test split & metrics
A time-based split should be used, where earlier months are used for training and more recent months for testing. A random split is inappropriate because it would introduce data leakage by mixing past and future observations.
Evaluation metrics such as RMSE and MAE should be used. RMSE penalizes large errors more heavily, making it useful for identifying significant prediction mistakes. MAE provides an easily interpretable average error in units of items sold. Together, they provide a comprehensive view of model performance.

(b) Explaining model decisions
Feature importance can be used to explain why different promotions are recommended for the same store in different months. For example, temporal features such as month or seasonal patterns may influence the effectiveness of certain promotions.
By analysing feature importance and model outputs, it becomes clear that customer behavior changes over time, and the model adapts its recommendations accordingly. This explanation can be communicated to the marketing team using simple visualisations and examples.    

(c) Deployment process
The trained model can be saved using serialization techniques such as joblib or pickle. Each month, new data is collected and processed using the same preprocessing pipeline before being fed into the model to generate predictions.
Monitoring should be implemented by tracking model performance over time using metrics such as RMSE. If performance degrades due to changes in customer behavior or market conditions, the model should be retrained with updated data.
This ensures that the system remains accurate and reliable in a real-world environment.