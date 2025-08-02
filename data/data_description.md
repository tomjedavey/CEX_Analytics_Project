# Data Description - Onchain Data from Flipside Crypto on Snowflake Marketplace + Features

## For the data with the directory: data/raw_data/new_raw_data_polygon.csv

### 1. Overview of the Dataset

Data is sourced from Flipside Crypto using SQL queries of Snowflake Marketplace to gain onchain crypto data. The data currently only holds onchain data for the Polygon blockchain, but as the base code is produced and progress is made overall in terms of producing a UEX, or the final output, then more data will be integrated from other blockchains to build a bigger picture and more value to CEXs.

The form of this supposed data from other blockchains will be decided and understood once significant process is made in terms of the production of the UEX, from there I can integrate a wider variety of blockchains to build a bigger picture for the purpose of providing analytic value.

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
**PRETTY MUCH ALL 0 VALUES**

20. **BRIDGE_OUTFLOW_COUNT**
    - Data Type: integer
    - Description: Number of bridge transactions moving assets out of Polygon

21. **BRIDGE_INFLOW_COUNT**
    - Data Type: integer
    - Description: Number of bridge transactions moving assets into Polygon
**DOESN'T WORK IN ITS CURRENT FORM**

22. **BRIDGE_TOTAL_VOLUME_USD**
    - Data Type: float
    - Description: Total USD volume of all bridge transactions (inflow and outflow combined)


### 3. Preprocessing Steps

Before modelling in the UMAP + Clustering, all features were log transformed and scaled. This is set in the config_cluster.yaml file and includes a feature to avoid log-transforming certain features given the context of what is trying to be achieved (this is seen with TX_PER_MONTH etc).

ADD THE FUNCTIONALITY FOR THE BELOW WHEN LOOKING TO PRODUCE THE UEX
Features engineered from the initial data produced though the SQL query are features engineered making use of the other features. This includes the likes of REVENUE_PROXY which takes the following formula:
__________

Key filtering within the data sampling:  **IN ITS CURRENT FORM**

- TX_PER_MONTH must be greater than or equal to 1.
- TOTAL_TRANSFER_USD must be greater than 0.
-  ACTIVE_DURATION_DAYS must be greater than or equal to 15.

The purpose of the above filters, is to utilise only what can be considered active wallets in terms of their analytic profiles seen in the UEX. **NEED TO ADD THIS CONTEXT IN FULLY WHEN COMING TO EVALUATE THE END OUTPUT TO BE DISPLAYED AND SHOWN TO PEOPLE + COMPANIES**

The functionality of the UMAP dimensionality reduction and HDBSCAN clustering pipeline also separates this data into two key clusters as a result of the current configurations seen in config_cluster.yaml. This data is available to be seen in the form of CSV files in the data/raw_data/cluster_datasets folder.

### 4. Data Sources and Access

All the onchain crypto data is sourced from the publically accessible tables from Flipside Crypto on Snowflake Marketplace. 
Access required a free trial account and use of credits

No API keys were utilised in the current version of this purpose (**CHANGE IF AUTOMATICALLY UPDATING IN THE FUTURE - FEEDBACK LOOP ETC**) - data was queried and produced using Snowflake Marketplace. The SQL query seen in query_new_raw_data.md took 10 minutes and 38 seconds to process and the data was downloaded as a CSV for use in this repository in its current functionality.

### 6. Versioning

The initial date of this data being used with the current functionality is from 16/01/2025 to 16/07/2025 (6 months). 

**IF THE GOAL IS TO PRODUCE A FEEDBACK LOOP UEX WITH AN AUTOMATIC DATA FLOW ETC - NEED TO UPDATE THE ABOVE ALONG WITH A NUMBER OF CASES IN THE END RESULT**

### 7. Directory Structure

data/
├── data_description.md
├── processed_data/
├── raw_data/
│ ├── raw_data/
│ │ ├── cluster_datasets/
│ │ │ ├── cluster_datasets_summary.JSON
│ │ │ ├── cluster_datasets_summary.txt
│ │ │ ├── new_raw_data_polygon_cluster_0.CSV
│ │ │ └── new_raw_data_polygon_cluster_1.CSV
│ │ ├── initial_raw_data_polygon.CSV
│ │ └── new_raw_data_polygon.CSV
├── reports/