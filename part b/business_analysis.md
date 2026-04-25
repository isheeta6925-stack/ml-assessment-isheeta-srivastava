# Part B — Business Case Analysis: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — Machine Learning Problem Formulation (3 marks)

**Target Variable:** `items_sold` — the number of items sold in a given store during a given month under a specific promotion.

**Candidate Input Features:**

| Category | Features |
|----------|----------|
| Store characteristics | `store_id`, `store_size` (small/medium/large), `location_type` (urban/semi-urban/rural), `competition_density` |
| Promotion | `promotion_type` (Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, Loyalty Points Bonus) |
| Temporal | `month`, `year`, `is_weekend`, `is_festival`, derived features such as `is_month_end` |
| Contextual | Monthly footfall estimates, customer demographic indicators (if available from store attributes) |

**Problem Type:** This is a **supervised regression** problem. The target variable `items_sold` is a continuous numerical quantity, not a category, so classification is inappropriate. We are learning a mapping from store-promotion-calendar features to a predicted sales volume using labelled historical observations — hence supervised, not unsupervised. Regression is the correct framing because the output is ordinal and continuous: a prediction of 320 items is meaningfully different from 280 in a way that a class label cannot capture.

---

### B1(b) — Why items_sold is a More Reliable Target Than Revenue (3 marks)

Revenue is the product of items sold, unit prices, and discount depth. When a Flat Discount promotion is active, every item contributes less revenue per unit — so revenue drops even if customer response (volume) is strong. Conversely, a premium product push can inflate revenue without any meaningful increase in footfall or promotion effectiveness. Using revenue as the target would cause the model to conflate pricing strategy with promotion effectiveness, making it impossible to isolate what the promotion itself is actually doing.

`items_sold` measures **customer behavioural response** directly — it answers the question "how many customers were moved to buy something as a result of this promotion?" — which is the actual business question. It is also less susceptible to price renegotiations, markdown accounting policies, and product-mix changes that vary across stores and months.

**Broader Principle:** This illustrates the principle of **target-objective alignment with noise minimisation** in real-world ML. The target variable must measure the phenomenon the business actually wants to optimise, stripped of as many confounding factors as possible. A model is only as meaningful as what it is trained to predict — if the target conflates two different causal mechanisms (volume response and price effects), the model learns a muddled signal and its recommendations will be unreliable. Choosing the right target is often more consequential than any modelling decision made afterwards.

---

### B1(c) — Alternative to a Single Global Model (2 marks)

A single global model trained on all 50 stores assumes that the relationship between promotion type and items sold is the same everywhere. In reality, a BOGO promotion in a dense urban store serving price-conscious commuters behaves very differently from the same promotion in a rural store where customers buy infrequently in large bulk trips. A global model will average these out and perform poorly on both.

**Proposed strategy: Location-stratified models with store-level fixed effects.** Concretely:

1. Train **separate models per location type** (urban, semi-urban, rural) — three models instead of one. Each model learns promotion response curves relevant to its customer base.
2. Within each model, include `store_id` as a feature (or as a fixed effect via store-level mean encoding) so that persistent store-level idiosyncrasies — local competition, demographics, physical layout — are captured without requiring a separate model per store.

This is an instance of the **hierarchical modelling** principle: share statistical strength across stores of the same type (avoiding overfitting to individual stores with few observations) while still respecting the structural differences between location types that a single global model cannot handle.

---

## B2. Data and EDA Strategy

### B2(a) — Table Joins and Data Grain (4 marks)

**Join Structure:**

The four source tables are joined sequentially using the following keys:

1. `transactions` LEFT JOIN `store_attributes` ON `store_id` — attaches physical and demographic store metadata (size, location type, footfall, competition density) to every transaction record. A left join ensures no transactions are lost if a store attribute record is missing.

2. Result LEFT JOIN `promotion_details` ON `promotion_type` AND `store_id` AND `month` — attaches promotion-specific metadata (start date, discount depth, eligibility criteria) to each transaction in the period when that promotion was active.

3. Result LEFT JOIN `calendar` ON `transaction_date` — attaches `is_weekend` and `is_festival` flags. The calendar table is a date-spine and every transaction date will have a match, so a left join is safe here.

**Grain of the final modelling dataset:** One row = one store × one calendar month × one promotion type. This is the prediction unit — the model is asked "given this store, this month, and this promotion, how many items will be sold?" Aggregations performed before modelling:

- `SUM(items_sold)` per store per month per promotion type → the target variable
- `AVG(basket_size)` per store per month → a contextual feature
- `COUNT(transactions)` per store per month → a proxy for footfall
- `AVG(competition_density)` per store (stable or slow-changing, can be taken from store attributes directly)

---

### B2(b) — EDA Strategy: Four Specific Analyses (4 marks)

**Analysis 1 — Distribution of items_sold (histogram + box plot)**

*What to look for:* Right skew (common in retail sales data), presence of extreme outliers, and whether the distribution differs meaningfully across location types. Specifically, check whether the variance is constant or whether high-sales stores show much wider spread.

*Influence on modelling:* If the distribution is heavily right-skewed, a log-transform of the target (`log1p(items_sold)`) may be appropriate to stabilise variance and prevent large observations from dominating the loss function. If outliers are found to correspond to festival days, they are real signal — not noise — and should be retained with `is_festival` as a feature.

**Analysis 2 — Boxplots of items_sold by promotion_type, stratified by location_type**

*What to look for:* Whether the median items_sold under each promotion type differs materially, and whether the ranking of promotion types is consistent across urban, semi-urban, and rural stores. For example, does BOGO outperform Flat Discount everywhere, or only in urban stores?

*Influence on modelling:* If promotion effectiveness varies by location type, an interaction feature `promotion_type × location_type` should be engineered. If one promotion consistently underperforms across all strata, it may be worth flagging to the marketing team before even building the model. This analysis is also the primary justification for or against a single global model.

**Analysis 3 — Time-series plot of monthly aggregate items_sold (line chart with festival markers)**

*What to look for:* Seasonal peaks (October–December festive season), year-over-year growth trend, and whether promotions are correlated with the peaks or are being deployed counter-cyclically. Also look for sudden drops that may indicate data quality issues (missed transactions, store closures).

*Influence on modelling:* Strong seasonality confirms that `month` and `is_festival` must be included as features. If there is a clear year-over-year upward trend, `year` should also be included. If promotions cluster around peaks (which they typically do), the model needs to disentangle promotion effect from base seasonal demand — this motivates including a seasonality baseline feature or using the calendar table's festival flags explicitly.

**Analysis 4 — Correlation heatmap of numerical features**

*What to look for:* Correlations between `competition_density` and `items_sold` (expected negative), between `store_size` proxy (e.g., footfall count) and sales, and between temporal features. Critically, check for multicollinearity between features that might be redundant (e.g., if `store_id` and `competition_density` are highly correlated because store assignment is deterministic by geography).

*Influence on modelling:* High multicollinearity between input features will not bias tree-based models but will destabilise Linear Regression coefficients. If detected, VIF (Variance Inflation Factor) analysis should be run and redundant features dropped before fitting any linear baseline. For Random Forest, correlated features dilute individual feature importance scores — this should be noted when interpreting the importance ranking.

---

### B2(c) — Addressing the 80% No-Promotion Imbalance (2 marks)

**How the imbalance affects the model:**

When 80% of the training rows represent "no promotion" conditions, the model has ten times more examples of baseline sales than promotional sales. A model minimising mean squared error across all rows will be heavily incentivised to fit the no-promotion distribution well and will underfit the promotional periods — the very cases we most want to predict accurately. The model may learn to predict close to the no-promotion mean for most inputs, producing poor recommendations when a promotion is actually active.

**Steps to address it:**

1. **Sample weights:** Assign higher `sample_weight` to rows where a promotion is active during model training (e.g., weight promotional rows by the inverse of their frequency, i.e., ×5). Most scikit-learn estimators accept `sample_weight` in their `fit()` call. This forces the model to fit promotional observations more carefully without discarding any data.

2. **Stratified evaluation:** When reporting RMSE and MAE, report them separately for promotional periods and non-promotional periods. A model that achieves good aggregate RMSE but poor RMSE on promotional rows is not fit for purpose — the disaggregated metrics will reveal this.

3. **Feature engineering:** Create an explicit binary flag `is_promotion` (1 if any promotion is active, 0 otherwise) to ensure the model can learn a clean boundary between promoted and non-promoted states. This makes the promotional signal explicit rather than relying on the model to infer it from `promotion_type` alone.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split Setup and Metrics (4 marks)

**Split Setup:**

With three years of monthly store-level data (36 months × 50 stores = 1,800 store-month records), the appropriate split is a **temporal holdout**:

- **Training set:** Months 1–30 (January 2022 – June 2024, roughly 83% of the timeline)
- **Test set:** Months 31–36 (July 2024 – December 2024, the most recent 6 months)

This mirrors real deployment: the model is trained on everything known up to a cutoff date, then evaluated on the months that follow — exactly as it will operate in production. If data volume permits, a **rolling-window cross-validation** (also called time-series cross-validation) should supplement the final holdout: train on months 1–24, validate on months 25–30; train on months 1–30, validate on months 31–36. This gives a more stable estimate of out-of-sample performance than a single split.

**Why a random split is inappropriate:**

A random split distributes records from 2022, 2023, and 2024 across both train and test sets. The model then trains on December 2024 data and is "tested" on January 2022 — a temporal reversal. The model effectively sees the future during training, producing inflated test metrics that will not replicate in production. More subtly, seasonal patterns that the model should learn from early years will leak directly into the test evaluation, making the model appear far more capable of generalising than it actually is.

**Evaluation Metrics and Business Interpretation:**

| Metric | Formula | Business Interpretation |
|--------|---------|------------------------|
| **RMSE** | √(mean of squared errors) | Penalises large errors disproportionately. An RMSE of 30 items means occasional predictions are off by much more than 30 — critical if the business uses predictions to set inventory or staff levels. A large RMSE signals the model is occasionally wildly wrong, which is operationally costly. |
| **MAE** | mean of absolute errors | The average prediction error in the same units as items_sold. An MAE of 20 means the model is typically off by 20 units per store per month — immediately interpretable to the marketing team. MAE treats all errors equally regardless of direction. |

Both metrics should be reported and compared. If RMSE >> MAE, there are occasional large individual errors that need investigation (e.g., a particular store during a festival month). If they are close, errors are relatively uniform across records.

---

### B3(b) — Communicating Model Logic Using Feature Importance (4 marks)

**Why the same store gets different recommendations in December vs March:**

After training, extract the Random Forest feature importances globally, then use a tool like **SHAP (SHapley Additive exPlanations)** to compute per-prediction, per-feature contribution values for Store 12's December and March predictions. SHAP decomposes each prediction into additive contributions from each feature, making the explanation local and exact rather than a global average.

**Concrete investigation for Store 12:**

For the December recommendation (Loyalty Points Bonus):
- `is_festival = 1` (December festive season) contributes a large positive SHAP value — festival months drive significantly higher baseline demand
- `month = 12` contributes positively — December is the peak month globally
- Under festival conditions, the model has learned that Loyalty Points Bonus retains premium customers who are already shopping more frequently — the marginal lift from Loyalty Points is highest when customers are already engaged

For the March recommendation (Flat Discount):
- `is_festival = 0` — no seasonal demand spike to leverage
- `month = 3` contributes negatively relative to December — March is a low-traffic month
- In low-traffic months, the model has learned that Flat Discount drives the largest volume response from price-sensitive customers who need an explicit incentive to visit

**Communicating to the marketing team:**

Present a simple table showing the top 3 contributing factors for each recommendation alongside their direction. Avoid statistical jargon — replace "SHAP value of +12.4" with "this feature increased our predicted items sold by approximately 12 units." A side-by-side bar chart of feature contributions for Store 12 in December versus March makes the seasonal logic immediately intuitive. The key message: the model is not being arbitrary — it is responding to the same signals the team would recognise (festival season drives retention promotions; quiet months drive acquisition discounts).

---

### B3(c) — End-to-End Deployment Process (4 marks)

**Step 1 — Saving the Model**

Serialise the entire scikit-learn `Pipeline` object (which includes the `ColumnTransformer` preprocessor and the trained model) using `joblib.dump(pipeline, 'promotion_recommender_v1.pkl')`. Saving the complete pipeline — not just the model weights — is critical: it ensures the identical preprocessing transformations (OHE category mappings, scaler mean and variance) are applied to new data at inference time. Store the serialised file in a version-controlled artefact repository (e.g., MLflow Model Registry or an S3 bucket with versioning enabled) alongside the exact training data cutoff date and the training metrics.

**Step 2 — Monthly Data Feed and Inference**

At the start of each month, the data engineering team runs a scheduled pipeline (e.g., an Airflow DAG or a cron job) that:

1. Extracts the previous month's transactions, store attributes, and calendar flags from the operational database
2. Aggregates to the store × month grain (as defined in B2a)
3. Engineers the same features used during training (year, month, day_of_week, is_month_end, is_festival)
4. Loads the saved pipeline: `pipeline = joblib.load('promotion_recommender_v1.pkl')`
5. Calls `pipeline.predict(X_new)` to generate items_sold predictions for each store × promotion_type combination
6. The combination with the highest predicted items_sold is returned as the recommended promotion for each store for the coming month

The output is a simple table: store_id | recommended_promotion | predicted_items_sold, delivered to the marketing team's dashboard.

**Step 3 — Monitoring and Retraining Triggers**

Monitoring operates on two levels:

- **Data drift monitoring:** Each month, compute the distribution of key input features (promotion mix, competition density, footfall proxy) and compare to the training distribution using the Population Stability Index (PSI). If PSI > 0.2 for any key feature, raise an alert — the input space has shifted and the model's learned relationships may no longer hold.

- **Prediction performance monitoring:** Each month, once actuals are available (after the month closes), compute the live RMSE and MAE by comparing the prior month's predictions to the actual items_sold figures. Maintain a rolling 3-month average. If the rolling RMSE exceeds the original test-set RMSE by more than **15%**, trigger a retraining review.

**Retraining cadence:** The model should be retrained at minimum every 6 months using the full updated historical data, regardless of whether the performance threshold is breached — customer behaviour and competitive dynamics in retail shift slowly but continuously. When retraining, the new model should be evaluated on the most recent 2 months of held-out data before replacing the production model, ensuring the update genuinely improves performance rather than just introducing noise from a small new data batch.
