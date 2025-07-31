# Data Description - Onchain Data from Flipside Crypto on Snowflake Marketplace + Features

## For the data with the directory: data/raw_data/new_raw_data_polygon.csv

### 1. Overview of the Dataset

Data is sourced from Flipside Crypto using SQL queries of Snowflake Marketplace to gain onchain crypto data. The data currently only holds onchain data for the Polygon blockchain, but as the base code is produced and progress is made overall in terms of producing a UEX, or the final output, then more data will be integrated from other blockchains to build a bigger picture and more value to CEXs.

The form of this supposed data from other blockchains will be decided and understood once significant process is made in terms of the production of the UEX, from there I can integrate a wider variety of blockchains to build a bigger picture for the purpose of providing analytic value.

- Date Range being used for building analytic scores - 16/01/2025 to 16/07/2025 (6 months)
- Size: 20,174 rows, 22 columns (21 features)
- File Type: CSV

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