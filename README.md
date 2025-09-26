# Producing analytic profiles of onchain crypto wallets to provide value in key decision making for Centralised Exchange Stakeholders
A project to produce a number of analytic scores on 20,000 model Polygon wallets to show example of value possible through this form of data analytics and science.

## README structure:

The following is the structure of this README documentation:

1. Introduction and Project Overview
2. Motivation and Business Relevance
3. Project Objectives
4. Repository Structure
5. Setup and Installation
6. Usage
7. Methods and Technical Details
8. Result and Insights
9. Key Learnings and Building Approach
10. Next Steps Possible With Actual Application
12. Acknowledgments and References
13. Link to the TLADS Canvases
14. License

## 1. Introduction and Project Overview

This project provides the neccessary processing and analytic score production for crypto wallets (making use of onachain crypto data) in order to allow for wallet segmentation based on behaviour. The purpose of this is in order to provide the opportunity for a Centralised Cryptocurrency Exchange (CEX) stakeholder to group wallets by behaviour and gain information relevant to the following business initative: "improve user retention and personalised engagement strategies for Centralised Crypto Exchanges (CEXs - user retention, churn prediction and behavioural segmentation) to increase fee revenue". This therefore forms the use case being looked at which can be summarised as "User segmentation based on wallet behavioural archetypes." 

The methods utilised to attempt to show an example of what this could look like in practice include a UMAP dimensionality reduction and HDBSCAN clustering pipeline utilised in activity clustering, along with other formulas to engineer analytic scores from the raw data. These analytic scores are simply a glimpse into the type of analytics intented, which would be available in practice with the correct levels of data (of course considering all other factors). The analytic scores built are labelled as follows and are explained in later parts of this documentation:

- Behavioural Volatility Score
- Cross Domain Engagement Score
- Revenue Proxy Score
- DEX events Interaction Mode Score
- CEX events Interaction Mode Score
- Bridge events Interaction Mode Score
- DeFi events Interaction Mode Score

## 2. Motivation and Business Relevance

Mentioned briefly in the introduction, the business relevance of this project (in its final form with integration of further data and more models and features) is in order to achieve improved user retention and engagement strategies for CEX stakeholders through the use case that is segmenting crypto wallets by behaviour. The full link to the business value from the methods produced in this repository will become clear when analytic scores and models are understood fully; but in terms of making this more clear in how it would work in practice, the wallet data would be linked to users of a CEX (of course with all privacy considerations included for) so that this project's output would build a valuable insight to a CEX's users. 

Of course, this project is not near actual application in this way. Therefore, building from this link to business relevance, my motivation in building this project was in order to show a glimpse of what is possible with this type of analytics and this methodology of thinking with use of the TLADs (Thinking Like a Data Scientist) methodology described by Bill Schmarzo (references later in this documentation). I built all the relevant TLADs canvases to develop my personal ways of thinking in regards to producing business value through data science and analytics, held in this repository also, and used these to sculpt the type of processes built in this project. This was completed in order to not only improve my understanding of this discipline and type of work, but also to display my understanding of how to go about building a project with valid connection to real-world analytics in practice.

## 3. Project Objectives

In terms of the objectives pursued in this project to build a valid analytic profile that described wallet behaviour, the following is a description of all the goals undertaken, of course a lot of these are not neccessarily the best methods to achieve the analytic value aspired to which is evaluated and explained later in this document: 

- Produce a clustering pipleine with the purpose of clustering wallets based on their general activity level. This involved the production of the UMAP dimensionality reduction (uniform maniford approximation and projection) and HDBSCAN clustering (hierarchical density based spatial clustering of applications with noise).
- Produce the neccessary functionality to preprocess and log-transform features being utilised in models and feature engineering of analytic scores.
- Produce an interpretable score that measures volatility of a wallet's transaction behaviour over time to form the behavioural volatility score.
- Produce an interpretable score that estimates the revenue a wallet provides in the given dataset's time period (of course something that would look different with integration of a CEX's data, the estimation proxy is only used to show how actual revenue data would be intergrated into an analytic profile).
- Produce an interpretable score that effectively measures a wallets interaction with a number of different event types in comparison to the dataset to form a interaction mode score.
- Produce an interpretable score that measures the diversity of interaction of a wallets interaction with different features, forming the cross domain engagement score.
- Produce the neccessary functionality to effectively visualise and display results of this project's output in a reproducible dashboard.

## 4. Repository Structure

CEX_Analytic_Project/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── INTERACTION_MODE_README.md
├── query_new_raw_data.sql
│
├── Notebooks/
│   └── Jupyter notebooks for EDA, clustering, and visualization
│
├── TLADS_Documents/
│   └── Business initiative and analytics documentation (Word files)
│
├── artifacts/
│   ├── Configurations_and_Pipeline_Metadata/   # YAML configs for pipeline and clustering
│   ├── Dashboards/                            # HTML dashboard outputs
│   └── Manual_Analysis_and_Configurations/     # Manual analysis reports and configs
│
├── data/
│   ├── data_description.md
│   ├── processed_data/
│   └── raw_data/
│
├── docs/
│   └── Markdown documentation for analytic modules and clustering
│
├── scripts/
│   ├── analytic_scores_merging_execution.py
│   ├── archive_execution_scripts/
│   ├── artifact_production_execution/
│   ├── behavioural_volatility_feature_engineering.py/
│   ├── clustering/
│   ├── cross_domain_/
│   ├── interaction_mode_score/
│   └── revenue_score/
│
├── source_code_package/
│   ├── __init__.py
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
│
└── tests/
    ├── __init__.py
    └── Unit tests for configuration, clustering, scoring, and feature engineering

## Author

[Tom Davey](https://github.com/tomjedavey)

**READ THE BELOW FOR ADDITIONAL NOTES ON HOW THE TLADS CANVASES RELATE TO THE ACTUAL MODELS BUILT IN THE PROJECT TO PRODUCE THE ANALYTICAL OUTPUT**

Note on Metrics and Business KPIs:
This project uses only onchain public data. However, the TLADS canvases include references to business metrics and KPIs that would typically be available within a centralized exchange (e.g., revenue per user, session frequency). These are included to develop and demonstrate the business-aligned thinking behind the modeling, even though they are not computed or used directly in the final models.

# **ONE THING TO POTENTIALLY THINK ABOUT WHEN COMING TO THE END / PROJECT OUTPUT - IS THE REVENUE CONTRIBUTION SCORE USELESS IF A CEX CAN ACTUALLY TRACK REVENUE GAINED PER USER?? - MAYBE CHANGE THIS FUNCTIONALITY BACK TO SOMETHING PREDICTIVE USING TIME-SERIES DATA FROM SNOWFLAKE WHERE POSSIBLE. NEED TO EVALUATE THIS TO A LARGE EXTENT**
