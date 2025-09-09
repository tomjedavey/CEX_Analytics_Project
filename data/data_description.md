# Data Description - Onchain Data from Flipside Crypto on Snowflake Marketplace + Features

## For the data with the directory: data/raw_data/new_raw_data_polygon.csv

### 1. Overview of the Dataset

Data is sourced from Flipside Crypto using SQL queries of Snowflake Marketplace to gain onchain crypto data. The data currently only holds onchain data for the Polygon blockchain for one 6 month period, but in terms of possibilities in a real world use case, of course a wide variety of blockchains would be utilised in addition to CEX user data to build a greater picture of user profiles to enhance key decision making. To add to this, utilisation of the Flipside Crypto API with snowflake marketplace could be used to add the ability for the user to run the analytic pipelines with configurations of their choice on up-to-date data in a form of self-serve analytics. More on the further possibilities with regards to the delivery of the analytic value available utilising the data pipeline built will be explained later when explaining the SQL query code used and how data flows link to this key area. 

The current version of the data being inputted to the various feature engineering and models for the purpose of a display of analytic value provision can be seen as follows:

- Date Range being used for building analytic scores - 16/01/2025 to 16/07/2025 (6 months)
- Size: 20,174 rows, 22 columns (21 features)
- File Type: CSV

Exploratory Data Analysis of this data for the purpose of evaluating its fit in use of analytic algorithms and more can be seen in the file with the directory: "Notebooks/initial_data_EDA.ipynb".

### 2. Schema and Field Descriptions

1. **WALLET**
   - Data Type: string
   - Description: Unique wallet address identifier for each user on the Polygon blockchain

2. **TX_PER_MONTH**
   - Data Type: float
   - Description: Average number of transactions per month for the wallet during the analysis period

3. **TOKEN_DIVERSITY**
   - Data Type: integer
   - Description: Number of different tokens the wallet has interacted with

4. **PROTOCOL_DIVERSITY**
   - Data Type: integer
   - Description: Number of different protocols the wallet has interacted with

5. **TOTAL_TRANSFER_USD**
   - Data Type: float
   - Description: Total USD value of all transfers made by the wallet

6. **INTERACTION_DIVERSITY**
   - Data Type: integer
   - Description: Number of different types of interactions the wallet has performed

7. **ACTIVE_DURATION_DAYS**
   - Data Type: integer
   - Description: Number of days the wallet has been active during the analysis period

8. **AVG_TRANSFER_USD**
   - Data Type: float
   - Description: Average USD value per transfer for the wallet

9. **USD_TRANSFER_STDDEV**
   - Data Type: float
   - Description: Standard deviation of USD transfer values, indicating transfer amount volatility

10. **DEX_EVENTS**
    - Data Type: integer
    - Description: Number of decentralized exchange (DEX) events/transactions

11. **GAMES_EVENTS**
    - Data Type: integer
    - Description: Number of gaming-related events/transactions

12. **CEX_EVENTS**
    - Data Type: integer
    - Description: Number of centralized exchange (CEX) events/transactions

13. **DAPP_EVENTS**
    - Data Type: integer
    - Description: Number of decentralized application (DApp) events/transactions

14. **CHADMIN_EVENTS**
    - Data Type: integer
    - Description: Number of chain admin events/transactions

15. **DEFI_EVENTS**
    - Data Type: integer
    - Description: Number of decentralized finance (DeFi) events/transactions

16. **BRIDGE_EVENTS**
    - Data Type: integer
    - Description: Number of bridge events/transactions for cross-chain transfers

17. **NFT_EVENTS**
    - Data Type: integer
    - Description: Number of non-fungible token (NFT) events/transactions

18. **TOKEN_EVENTS**
    - Data Type: integer
    - Description: Number of token-related events/transactions

19. **FLOTSAM_EVENTS**
    - Data Type: integer
    - Description: Number of miscellaneous or uncategorized events/transactions

20. **BRIDGE_OUTFLOW_COUNT**
    - Data Type: integer
    - Description: Number of bridge transactions moving assets out of Polygon

21. **BRIDGE_INFLOW_COUNT**
    - Data Type: integer
    - Description: Number of bridge transactions moving assets into Polygon

22. **BRIDGE_TOTAL_VOLUME_USD**
    - Data Type: float
    - Description: Total USD volume of all bridge transactions (inflow and outflow combined)


### 3. Data Preprocessing Steps

Before modelling in the UMAP + Clustering, all features were log transformed and scaled. This is set in the config_cluster.yaml file and includes a feature to avoid log-transforming certain features given the context of what is trying to be achieved (this is seen with TX_PER_MONTH etc).

Analytic scores are engineered from the initial data produced though the SQL query are engineered making use of the other raw data features, this in cases includes statistical models and complex processes. These analytic scores are derived as follows, explanations of rationale and insight available will be completed in other documentation (README):

- REVENUE_PROXY = 0.4 * AVG_TRANSFER_USD * TX_PER_MONTH
                + 0.35 * (DEX_EVENTS + DEFI_EVENTS) * AVG_TRANSFER_USD
                + 0.25 * BRIDGE_TOTAL_VOLUME_USD

- BEHAVIOURAL_VOLATILITY_SCORE = Weighted composite of three components. 35% of this is equal to USD_TRANSFER_STDDEV / AVG_TRANSFER_USD. 40% of this is a composite of coeffient of variance (0.4 weighting), variance ratio (0.3 weighting) and Gini coefficient (0.3 weighting) across event types (EVENT count columns in raw data). And finally, the last 25% of this is a measure of diversity of activity, calculated as the average diversity.

- CROSS_DOMAIN_ENGAGEMENT-SCORE = -Σ (p_i * log₂(p_i)) / log₂(N). This is where p_i is the proportion of events in domain i and N is the number of event types with activity. This produces a normalised shannon entropy of event type proportions.

- INTERACTION_MODE_SCORES = Separate run of the UMAP + HDBSCAN clustering algorithm on the main raw data and activity-based clustered datasets. This is done with separate configurations to activity clustering on event features - DEX_EVENTS, CEX_EVENTS, DEFI_EVENTS and BRIDGE_EVENTS. From the results of these clusterings and for each feature, the cluster with the highest median non-zero value and sufficient activity is taken and from there a normalised distance score is produced for each wallet's features from the selected clusters' feature median values.

More on this will be explained when looking on this in the README and various other relevent documentation. Of course there are a lot more details with regards to normalisation of values in feature production, configuration options and more; this will all be explained in other documentation. Activity based clustering using the UMAP dimensionality reduction (uniform manifold approximation and projection) and HDBSCAN clustering (hierarchical density based spatial clustering of applications with noise) also plays a great role in describing analytic profiles. This is not an analytic score of the same form as the others however, this is explained in greater details later in this section and other documentation.

Key filtering within the data sampling is explained below. These are the important filtrations input to the SQL query (touched upon below) that produces a certain demographic of wallets in the raw onchain crypto data to enhance analytic value provision to CEX stakeholders:  

- TX_PER_MONTH must be greater than or equal to 1.
- TOTAL_TRANSFER_USD must be greater than 0.
- ACTIVE_DURATION_DAYS must be greater than or equal to 15.
- Wallets are also grouped and sampled by transaction count. 

The purpose of the above filters is to utilise only what can be considered active wallets in terms of their analytic profiles seen in the UEX. In addition to this, the grouping and sampling by transaction count (utilising a limit of 10,000 wallets below 10, 20,000 wallets between 10-99, and 20,000 wallets above 100 all in terms of transaction count) is to ensure that the data produced shows a strong distribution of wallet types to be segmented. In the case of this being put into actual use, of course you wouldn't have this pre-grouped data. However, given the understanding that this project is designed as a minimum viable product, this is done to show a strong distribution of data and possible wallet behaviours without taking on the larger amounts of data that further infrastructure would be needed for if this were to actually be in use. As long as this key point is understood when looking at interpretations and insights, value provision can be understood clearly.

These filters can be seen to be put in action in the query_new_raw_data.sql file in this repository which is the SQL query used to gain the raw data (data/raw_data/new_raw_data_polygon.csv). This SQL query is built for implementation into Snowflake Marketplace to gain a CSV file of the data being used initially. If this project was to be developed further and used in addition to user data to produce actual value to a CEX (instead of a display of potential analytic value), this SQL query would have to be developed to be used in an API call to Flipside Crypto to gain larger amounts of data on a wider range of blockchains that can automatically update periodically to provide stakeholders with the ability to update their own analytics (a form of self-serve analytics potential).

The functionality of the UMAP dimensionality reduction and HDBSCAN clustering pipeline also separates this data into two key clusters as a result of the current configurations seen in config_cluster.yaml. This data is available to be seen in the form of CSV files in the data/raw_data/cluster_datasets folder and is used not only to provide more insight into wallet behviour but also set up further analytic score engineering on this clustered data seen in the interaction mode score productions. Configurations of this pipeline of course make a massive impact on the insight available here; the ability to change this allows for strong reproducibility of the pipeline and adaptability to new data to provide greater scope of analytics.

### 4. Data Sources and Access

All the onchain crypto data is sourced from the publically accessible tables from Flipside Crypto on Snowflake Marketplace. 
Access required a free trial account and use of credits, in the case of integrating a data flow as previously mentioned, this would involve greater Snowflake usage and as a result fee charges.

No API keys were utilised in the current version of this project (as mentioned previously) - data was queried and produced using Snowflake Marketplace. The SQL query seen in query_new_raw_data.md took 10 minutes and 38 seconds to process and the data was downloaded as a CSV for use in this repository in its current functionality. Changes would need to be made to the query to accomodate more advanced data flow in potential developed iterations of this project as mentioned in the above section.

### 5. Data Quality & Limitations

Filters explained in the "Data Preprocessing Steps" section have enabled largely high quality data and avoided overcrowding of data with extremely-low activity wallets / data points. However, this results in some limitations in raw data columns such as the likes of BRIDGE_INFLOW_COUNT which displays non-usable results as well as FLOTSAM_EVENTS which produces all zero values as a result of the filters on the types of wallets being looked at (wallets more applicable to analytics for a CEX). This doesn't damage production of models and analytic score engineering in the data pipelines but is worth noting in terms of interpretations and explorations of the raw data itself as a potential limitation. 

In addition to this, raw onchain crypto data of course lacks the level of insight on its own to produce strong recommendations to the key decisions of a CEX stakeholder. In potential further developments of this project, the integration of CEX user data would provide much stronger analytics of wallet behaviour which shows that this data and the analytic scores and profiles built on it in its current form is limited in impact. This does not mean that this project serves no purpose however, as a display of the ways in which analytic value can be provided to a CEX this is a strong sight at what could be possible. It is just important to note this and understand well what the potential growth of this type of data science / analytics could provide.

### 6. Data Privacy & Ethics

All data utilised in this repository is public blockchain data. No personally identifiable information is used, which is what would be needed in terms of the evolution of this type of a project to start providing actual actionable insight. Ethical considerations would need to be made at later stages if a concept such as that built in this repository was to be undertaken. This would be things such as not typing wallets to indenties without consent. However, in terms of the value still available taking these potential ethical considerations into account, to understand trends in wallets and CEX users without actioning based on knowing who specifically owns each wallet or which user is linked to which wallet still provides strong opportunity to guide decision making based on changing customer bases, churn rates and more. 

### 7. Versioning

The initial date of this data being used with the current functionality is from 16/01/2025 to 16/07/2025 (6 months). 

### 8. Directory Structure

**STILL NEED TO UPDATE THE BELOW WHEN COMING TO PRODUCE THE ARTIFACTS ETC USED IN THE OUTPUT MAYBE??**
data/
├── data_description.md
├── processed_data/
│   ├── behavioral_volatility_features.csv
│   ├── cross_domain_engagement_features.csv
│   ├── revenue_proxy_features.csv
│   ├── cluster_datasets/
│   │   ├── cluster_datasets_summary.json
│   │   ├── cluster_datasets_summary.txt
│   │   ├── new_raw_data_polygon_cluster_0.csv
│   │   └── new_raw_data_polygon_cluster_1.csv
│   ├── clustering_results/
│   │   ├── cluster_labels.csv
│   │   ├── clustered_data.csv
│   │   ├── clustering_summary.txt
│   │   └── hdbscan_clustering_results.png
│   └── interaction_mode_results/
│       ├── cluster_0_clustering/
│       │   ├── cluster_labels.csv
│       │   ├── clustered_data.csv
│       │   ├── clustering_summary.txt
│       │   ├── full_absolute_distances.csv
│       │   ├── full_normalized_distances.csv
│       │   ├── full_raw_distances.csv
│       │   ├── full_weighted_distances.csv
│       │   └── hdbscan_clustering_results.png
│       ├── cluster_0_clustering_feature_medians.csv
│       ├── cluster_1_clustering/
│       │   ├── cluster_labels.csv
│       │   ├── clustered_data.csv
│       │   ├── clustering_summary.txt
│       │   ├── full_absolute_distances.csv
│       │   ├── full_normalized_distances.csv
│       │   ├── full_raw_distances.csv
│       │   ├── full_weighted_distances.csv
│       │   └── hdbscan_clustering_results.png
│       ├── cluster_1_clustering_feature_medians.csv
│       ├── interaction_mode_cluster_selections.json
│       ├── interaction_mode_cluster_selections.yaml
│       ├── interaction_mode_pipeline_summary.txt
│       ├── main_clustering/
│       │   ├── cluster_labels.csv
│       │   ├── clustered_data.csv
│       │   ├── clustering_summary.txt
│       │   ├── full_absolute_distances.csv
│       │   ├── full_normalized_distances.csv
│       │   ├── full_raw_distances.csv
│       │   ├── full_weighted_distances.csv
│       │   └── hdbscan_clustering_results.png
│       └── main_clustering_feature_medians.csv
├── raw_data/
│   └── new_raw_data_polygon.csv