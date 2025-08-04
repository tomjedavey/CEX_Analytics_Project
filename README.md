**NEED TO UPDATE THE FOLLOWING TO FILL OUT EACH OF THE BELOW SECTIONS ONCE THE NECCESSARY PROGRESS HAS BEEN MADE (LIKELY POST THE MODEL FUNCTIONALTIY BUILT IN WHOLE AND PRE END OUTPUT/UEX). MAKE SURE IN AREAS TO INTEGRATE THE EXPLORATION COMPLETED MAKING USE OF THE TLADS DOCUMENTS - THIS IN TERMS OF MAKING SURE DESCRIPTION OF THE PROJECT IS DONE FACING THE VALUE PROVISION TO A CEX WITH IDENTIFIED BUSINESS INITIATIVE AND USE CASES**

# Producing analytic profiles of onchain crypto wallets to provide analytic value for Centralised Exchanges


## Project Overview

Objective:
This project applies the Thinking Like a Data Scientist (TLADS) methodology to design and build a wallet-level segmentation framework for Centralized Crypto Exchanges (CEXs). The aim is to enhance user retention, personalize engagement strategies, and ultimately increase fee-based revenue through nanoeconomics-driven decision-making.

Use Case Focus:
The core use case is wallet segmentation based on behavioral archetypes derived from on-chain activity. By creating analytic scores that quantify user behaviors and propensities, CEXs can identify strategic cohorts (e.g., high-LTV users, dormant accounts, product explorers) and tailor retention, UX, and incentive mechanisms accordingly.

LTV Prediction Score
**(More to be confirmed during model development)**

Data:
The project uses granular wallet activity data from the Polygon blockchain, sourced via Flipside Crypto’s polygon.core schema. This includes:
**UPDATE WHEN MOVING ON FROM JUST POLYGON BLOCKCHAIN DATA WITH POTENTIAL AUTOMATED DATA INTEGRATION INTO THE MODEL PIPELINES**
📄 **[INSERT data types] — e.g., transaction records, contract interactions, value transfers**

📅 **[INSERT date range or snapshot period]**

🔍 **[INSERT any preprocessing or filtering rules]**

Output:
The segmentation pipelines with produce an output with the following presentation:

🧠 **Per-wallet analytic scores (e.g., LTV, innovation, volatility)**

🧩 **Cluster labels representing behavioral archetypes**

📊 **Dashboard-ready insights for strategy teams at CEXs**

**NEED TO FULLY UPDATE ALL OF THE ABOVE WHEN GETTING TO THE ACTUAL OUTPUT SECTION - UEX FEEDBACK LOOP ETC...**

CEXs could use this output to:

Prioritize incentive programs and A/B test offers

Route users to different UX paths based on complexity preference

Identify retention risks early and deploy interventions

Target high-value users for premium services

Methodology Alignment:
The project follows the TLADS 8-step framework, particularly emphasizing:

Entity-level nanoeconomic insights

Score transparency and interpretability

Algorithm selection based on stakeholder value and explainability

••IN THIS SECTION AT THE END - ONCE EVERYTHING FULLY COMPLETED CODE WISE ETC - NEED TO BE VERY SPECIFIC ABOUT HOW THE PROJECT'S OUTPUT FOLLOWS THE BUSINESS VALUE BASED APPROACH DESCRIBED IN THE TLADS BOOK (ALSO EXPLAINING THE SELECTED DATA STACK AND ALGORITHMS ALONG WITH THIS RATIONALE)**

## Key Features

- UMAP dimensionality reduction into a HDBSCAN clustering algorithm to segment wallets based on activity level
-
**NEED TO ADD IN HERE ALL THE ANALYTIC SCORES AND MODELS BUILT TO PRODUCE THE ANALYTICAL OUTPUT**

## Testing & CI/CD

**DO ONCE GITHUB ACTIONS, PYPROJECT.TOML AND RELATED TESTING FOLDER ALL UPDATED AND RUNNING AS NEEDED**

## Getting Started

**INSTRUCTIONS FOR HOW TO RUN AND USE THIS REPOSITORY TO GAIN THE INTENTED OUTPUT ETC - MAYBE NOTES ABOUT THE WAY PEOPLE NEED TO USE IT TO LOOK AT CERTAIN DATA ETC**

## Project Structure

**FULL REPOSITORY STRUCTURE - DO WITH ALL NEW FOLDERS FOR THE MODELS TO BE BUILT**

## Results

**NEED TO RESEARCH WHAT TO PUT HERE MORE - MAYBE SCREENSHOTS OF INTENDED OUTPUT GIVEN CURRENT CONFIGURATIONS FOR THE PEOPLE USING THE REPO**

## References

Data sourced from Flipside Crypto's 'Polygon.core' schema.
**IF AUTOMATED DATA INTEGRATION USED ETC -- UPDATE ACCORDINGLY BASED ON THE FRAMING OF THIS PROJECT AND HOW THE DATA FLOW IS CHANGED IN RELATION**

## Author

[Tom Davey](https://github.com/tomjedavey)

**READ THE BELOW FOR ADDITIONAL NOTES ON HOW THE TLADS CANVASES RELATE TO THE ACTUAL MODELS BUILT IN THE PROJECT TO PRODUCE THE ANALYTICAL OUTPUT**

Note on Metrics and Business KPIs:
This project uses only onchain public data. However, the TLADS canvases include references to business metrics and KPIs that would typically be available within a centralized exchange (e.g., revenue per user, session frequency). These are included to develop and demonstrate the business-aligned thinking behind the modeling, even though they are not computed or used directly in the final models.