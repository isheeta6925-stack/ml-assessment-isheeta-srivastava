Retail & Health Analytics: End-to-End Machine Learning Assessment
📌 Project Overview
This project is a deep dive into three distinct areas of Data Science: diagnostic classification, behavioral customer segmentation, and retail sales forecasting. Beyond just writing code, the goal was to build production-ready pipelines and provide strategic business insights that move the needle for stakeholders.
📂 Repository Guide
•	part_a/: The "Engine Room."
o	q1_supervised.ipynb: Predicting heart disease. Focuses on handling clinical data imbalances and clinical sensitivity (Recall/F1).
o	q2_unsupervised.ipynb: Clustering 500+ customers. Uses PCA to transform raw behavior into visual, actionable segments.
o	q3_feature_engineering.ipynb: A robust regression pipeline for sales. Features custom temporal engineering and a leakage-proof time-series split.
•	part_b/: The "Boardroom."
o	business_analysis.md: Translating metrics into money. A strategic look at how to deploy these models and monitor them for "model drift" in the real world.
•	data/: Cleaned datasets used across all modules.
________________________________________
🛠️ Key Technical Decisions
1. The "Clinical First" Approach (Supervised)
In the Heart Disease model, I prioritized the F1-Score over simple Accuracy. In a medical context, a "False Negative" (missing a sick patient) is much more dangerous than a "False Positive." The Random Forest was tuned specifically to balance these stakes.
2. Segmenting with Signal, Not Noise (Unsupervised)
Raw customer data is often messy. I used Principal Component Analysis (PCA) to strip away the noise and visualize our segments on two axes: Overall Value and Engagement Recency. This allows the marketing team to target "at-risk" high-spenders differently than "active" budget shoppers.
3. Avoiding the "Look-Ahead" Trap (Regression)
For the retail data, a random 80/20 split is a mistake because it lets the model "peek" into the future. I implemented a Temporal Split, training on the past and testing on the most recent 20%, ensuring the model is evaluated on how it would perform in a real-world "next month" scenario.
