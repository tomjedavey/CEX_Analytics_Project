# Data Description - Onchain Data from Flipside Crypto on Snowflake Marketplace + Features

Data is sourced from Flipside Crypto using SQL queries of Snowflake Marketplace to gain onchain crypto data. The data currently only holds onchain data for the Polygon blockchain, but as the base code is produced and progress is made overall in terms of producing a UEX, or the final output, then more data will be integrated from other blockchains to build a bigger picture and more value to CEXs.

The data shows a number of pre-engineered features (completed on SQL and as can be seen in the query under "query_initial_data.sql") for each wallet address over the 6 month period before the 4th July. This data includes 50,000 rows and subsequently data for 50,000 wallets on the Polygon blockchain.

The features engineered are as follows:

1. transaction frequency - transactions per week/month
2. token diversity - number of unique tokens used
3. protocol diversity - number of unique contracts interacted with
4. total transfer diversity - sum of token values sent/recieved
5. interaction diversity - number of distinct methods used (e.g., swap, stake etc)
6. active duration - number of days between first and last activity
7. Average transfer USD
8. DEX events count 
9. Dapp events count
10. Chadmin events count
11. DeFi events count
12. Bridge Events count
13. NFT events count
14. Token events count
15. Flotsam events count
16. Bridge outflow events count
17. Bridge inflow events count
18. Bridge total volume USD


#### Key to remember here - more features have to be engineered or maybe the current features may have to be normalised or tweaked in python code in order to produce the analytic scores (using analytic algorithms of some sort) successfully.