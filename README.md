# Retail Customer Intelligence Platform

## Project Overview

Retail Customer Intelligence Platform is an end-to-end analytics and machine learning project built on the UCI Online Retail dataset. The project studies customer behavior through transaction history, invoice structure, product descriptions, geographic distribution, and purchase timing, then turns those observations into practical decision support for segmentation, retention, future customer value prioritization, personalization, and demand monitoring.

The repository is structured as a reusable case study rather than a single notebook. Shared logic lives in `src/`, the notebook tells a clean analytical story, and the Streamlit application exposes the same outputs through an interactive interface.

## Executive Summary

### Key findings

- Customer segmentation provides a practical way to distinguish high-value loyal buyers, lower-engagement accounts, and more seasonal or emerging customer groups using recency, spend, basket, and diversity signals.
- Cohort and retention analysis highlights how repeat purchasing evolves after first purchase and helps surface which acquisition periods appear to sustain revenue more effectively over time.
- The future customer value module converts historical behavior into a near-term prioritization view, using rolling snapshots to identify customers more likely to spend or return within the next 90 days.
- Recommendation and demand modules complement the customer view by comparing generic versus personalized product suggestions and by tracking which product families show more persistent or shifting demand patterns.

### Business implications

- Customer treatment can be made more targeted by aligning messaging, cadence, and product emphasis to distinct behavioral segments rather than treating the full base uniformly.
- Early retention monitoring matters because cohort patterns often reveal value decay quickly, making it easier to focus intervention on weaker customer vintages before inactivity becomes entrenched.
- Near-term value scoring creates a defensible way to prioritize retention and reactivation activity when resources or marketing attention need to be allocated selectively.
- Product-family demand and recommendation outputs provide a more connected planning view across CRM, merchandising, and assortment decisions, even without a full production recommendation stack.

## Motivation

Retail decision-making is shaped by timing, repeat purchase behavior, product mix, and heterogeneous customer preferences. A transaction table can answer narrow descriptive questions, but it becomes much more useful when treated as a customer intelligence system:

- Which customers are building durable value versus buying occasionally?
- Which acquisition cohorts retain and monetize better over time?
- Which customers are most promising over the next 90 days?
- Does simple personalization outperform generic bestseller recommendations?
- Which product families show the strongest demand concentration, seasonality, and recent momentum?

The goal is not to overstate what a public dataset can do. The goal is to demonstrate a disciplined, business-aware workflow that extracts credible signals from observed customer-product interactions.

## Objectives

The platform is organized into six analytical modules:

1. Customer segmentation using behavioral and transactional features.
2. Cohort and retention analysis using first-purchase month.
3. Future customer value modeling using rolling customer snapshots.
4. Product recommendations with both baseline and personalized methods.
5. Demand and trend analysis at the product-family level.
6. Final synthesis that turns outputs into practical retail interpretation.

## Business Questions and Analytical Approach

| Business Question | Analytical Approach | Primary Output |
|---|---|---|
| Which customers appear most valuable or strategically important? | Behavioral feature engineering and unsupervised segmentation using recency, spend, basket depth, diversity, and repeat signals | Customer segments, segment profiles, sample customer views |
| How does retention vary across acquisition cohorts? | First-purchase cohort construction with monthly retention and cohort revenue tracking | Retention matrix, cohort revenue curves, early-repeat patterns |
| Which customers should be prioritized for near-term value? | Rolling snapshot modeling for 90-day future spend or repeat purchase propensity | Customer-level prediction scores and model evaluation tables |
| Does personalization outperform generic recommendations? | Offline comparison of a popularity baseline against item-item similarity recommendations | Precision@k, recall@k, hit rate, sample recommendation outputs |
| Which product families are gaining or losing momentum? | Temporal aggregation of revenue and units by derived product family with trend and timing analysis | Demand trend charts, category shift tables, short-horizon forecasts |

## Dataset

This project uses the **UCI Online Retail** dataset, a transaction-level retail dataset containing invoice records, stock codes, product descriptions, quantities, prices, customer identifiers, timestamps, and country.

### Expected file placement

Place the raw file in `data/raw/` or a subfolder beneath it. The loader searches recursively for common filenames such as:

- `Online Retail.xlsx`
- `online_retail.xlsx`
- `Online Retail.csv`
- `online_retail.csv`

The current implementation is designed around the original UCI dataset file, typically distributed as an Excel workbook.

## Development Strategy

The UCI dataset is light enough that the default configuration loads the full file rather than forcing a reduced sample. The code still supports customer-level sampling for development experiments or memory-constrained environments.

The platform standardizes the raw data into a reusable schema with:

- `order_id`
- `product_id`
- `product_name`
- `product_family`
- `customer_id`
- `country`
- `quantity`
- `unit_price`
- `revenue`
- transaction date and calendar features

It also removes cancellations and non-positive quantity or price records so the downstream models focus on realized purchases.

## Project Structure

```text
.
|-- app
|   |-- Overview.py
|   |-- pages
|   |   |-- 1_Customer_Segmentation.py
|   |   |-- 2_Cohort_&_Retention.py
|   |   |-- 3_Future_Customer_Value.py
|   |   |-- 4_Recommendations.py
|   |   |-- 5_Demand_Trends.py
|   |   `-- 6_Key_Findings.py
|-- data
|   |-- processed
|   `-- raw
|-- figures
|-- models
|-- notebooks
|   `-- retail_customer_intelligence_platform.ipynb
|-- scripts
|   |-- generate_notebook.py
|   |-- prepare_data.py
|   `-- train_models.py
|-- src
|   |-- config.py
|   |-- pipeline.py
|   |-- data
|   |-- features
|   |-- models
|   `-- visualization
|-- project_bullets.txt
|-- requirements.txt
`-- README.md
```

## Methodology

### 1. Data cleaning and preparation

The preparation pipeline:

- standardizes the UCI raw schema into internal retail-friendly field names
- removes cancellations and invalid price or quantity rows
- derives month, season, weekday, and weekend indicators
- derives a lightweight `product_family` label from product descriptions for category-style analysis
- saves processed tables for downstream analysis and app consumption

Prepared outputs are written to `data/processed/`, and model artifacts are stored in `models/`.

### 2. Feature engineering

Customer-level features are deliberately retail-oriented rather than limited to a basic RFM template. The feature set includes:

- recency
- order count, line count, and unit count
- total spend
- average order value
- average and maximum basket units
- average distinct products per order
- product-family diversity
- repeat product rate
- repeat purchase rate
- average interpurchase time
- spend concentration across product families
- dominant-country concentration
- weekend order share
- seasonal concentration and entropy
- tenure and spend-per-day normalization

These features support both segmentation and predictive modeling.

## Segmentation Approach

The segmentation module compares:

- `KMeans`
- `Gaussian Mixture Models`

Both operate on scaled behavioral features. The project evaluates candidate solutions across a small cluster range and compares them using silhouette score, with GMM also exposing BIC for additional model-selection context.

Once a preferred solution is chosen, clusters are profiled using spend, order count, basket depth, recency, diversity, and repeat behavior. The project then assigns intuitive segment names such as:

- High-Value Loyalists
- Basket Builders
- Recent Emerging Buyers
- Seasonal Buyers
- Low-Engagement Accounts

The exact labels depend on the fitted cluster characteristics in the prepared data.

## Cohort Analysis Approach

Retention analysis is built from each customer’s first purchase month. The cohort pipeline calculates:

- monthly cohorts
- month index since first purchase
- retained customers by cohort period
- retention rate tables
- revenue by cohort over time

This reveals whether some cohorts drop sharply after month 0, whether some retain more steadily, and whether monetization is concentrated in acquisition month or persists across follow-on periods.

## Predictive Modeling Approach

### Target selection

The project does not hard-code a future customer value target without checking the observed label distribution. Instead, it builds rolling customer snapshots and inspects target density before choosing the primary task.

The default recommendation logic is:

- use **90-day future spend** when the positive-spend share is sufficiently dense for a stable regression setup
- otherwise fall back to **repeat purchase in the next 90 days**

This keeps the prediction problem grounded in what the dataset can support instead of forcing an unstable target.

### Snapshot construction

For each cutoff date:

- features are built from the previous 180 days of observed history
- labels are built from the following 90 days
- snapshots are stacked into a time-aware modeling dataset

### Models

The project trains:

- a simple baseline model
- a stronger non-linear model based on histogram gradient boosting

If the chosen task is classification, logistic regression is also included as an interpretable benchmark.

### Evaluation

Metrics depend on the selected task:

- regression: RMSE, MAE, R²
- classification: ROC-AUC, PR-AUC, F1, precision, recall

The split is temporal rather than random, using earlier snapshots for training and later snapshots for evaluation.

## Recommendation Approach

The recommendation module includes two approaches:

1. **Popularity baseline**  
   Recommends the most purchased products not yet seen by the customer.

2. **Personalized item-item similarity**  
   Builds a customer-product interaction matrix and uses cosine similarity between products to recommend items related to previously purchased products.

### Offline evaluation

The recommender is evaluated using a holdout period from the end of the transaction history. Metrics include:

- precision@k
- recall@k
- hit rate

This is intentionally a pragmatic recommender rather than a production-grade ranking stack. It is sufficient to demonstrate whether purchase-history personalization adds value beyond generic popularity.

## Demand Analysis

The demand module focuses on business-relevant temporal patterns rather than full-store forecasting. It includes:

- monthly product-family demand
- revenue, units, and customer counts by product family
- recent product-family shifts
- weekday and month timing patterns
- a simple short-horizon demand forecast for selected product families

Because the UCI dataset does not provide an explicit product hierarchy, the platform derives `product_family` heuristically from product descriptions. That derived grouping is useful for exploratory analysis, but it should not be mistaken for an official merchandise taxonomy.

## Streamlit Application

The app is organized into these pages:

- Overview
- Customer Segmentation
- Cohort & Retention
- Future Customer Value
- Recommendations
- Demand Trends
- Key Findings

The interface is artifact-driven, which keeps it responsive once data preparation and model training are complete.

## Key Findings the Platform Is Designed to Surface

Once the dataset is loaded and the pipeline runs, the project is designed to answer questions such as:

- which segments contain the highest-value and most loyal customers
- whether recent cohorts are improving or weakening in early retention
- which customers warrant near-term value-focused prioritization
- whether item-history recommendations beat broad bestseller lists
- which product families are gaining or losing momentum over time

The exact numeric findings depend on the prepared data and should be interpreted in context rather than treated as universal retail truths.

## Limitations and Assumptions

- The dataset is transaction-level retail data, so the analysis is strongest on observed purchase behavior rather than broader customer journey context.
- There is no browsing, session, or clickstream data, which limits the ability to study consideration behavior, discovery paths, or on-site intent before purchase.
- There is no campaign exposure or attribution data, so changes in purchasing cannot be tied credibly to marketing touchpoints or channel-specific uplift.
- There is no inventory state or product-page interaction data, so recommendation and demand outputs are not inventory-aware and do not reflect availability constraints or product interest without purchase.
- Product-family labels are derived heuristically from product descriptions, which is useful for analysis but not equivalent to a governed merchandise taxonomy.
- Conclusions should be interpreted as directional and analytical rather than as production-grade causal claims or fully operational decision rules.

### Future Extensions

- Add web analytics events such as sessions, product views, cart actions, and search behavior to improve customer intent modeling and pre-purchase analysis.
- Incorporate merchandising data such as assortment structure, hierarchy, and brand or supplier attributes to strengthen category and mix analysis.
- Add promotion and markdown data to separate underlying demand from price-driven purchasing behavior.
- Integrate inventory and availability data so demand signals and recommendations can be evaluated in an operational context.
- Replace heuristic product-family derivation with a richer product taxonomy or master data layer.
- Add online experimentation or holdout-test data to support stronger causal evaluation of recommendations, retention interventions, and prioritization strategies.

## How To Run

### 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Place the dataset

Copy the UCI Online Retail file into `data/raw/`.

### 3. Generate processed data and train models

```bash
python scripts/train_models.py
```

This runs data preparation, creates processed tables, builds segmentation outputs, trains the future value models, and fits the recommendation artifacts.

### 4. Generate the notebook file

```bash
python scripts/generate_notebook.py
```

### 5. Launch the Streamlit app

```bash
streamlit run app/Overview.py
```

## Notes on Interpretation

This repository is about a customer intelligence platform and its value comes from joining multiple views of the business:

- segments explain heterogeneity
- cohorts explain lifecycle durability
- predictive modeling prioritizes who matters next
- recommendations personalize what to show
- demand analysis explains where product-family momentum is shifting

That combined view is the main point of the project.


## Author
**Shwetha Tinnium Raju**
