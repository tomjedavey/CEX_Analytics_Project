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
11. Acknowledgments and References
12. Link to the TLADS Canvases
13. License

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

CEX_Analytics_Project/
├── README.md
├── LICENSE
├── pyproject.toml
├── INTERACTION_MODE_README.md
├── query_new_raw_data.sql

├── Notebooks/
│   └── Jupyter notebooks for EDA, clustering, and visualization

├── TLADS_Documents/
│   └── Business initiative and analytics documentation (Word files)

├── artifacts/
│   ├── Configurations_and_Pipeline_Metadata/   # YAML configs for pipeline and clustering
│   ├── Dashboards/                            # HTML dashboard outputs
│   └── Manual_Analysis_and_Configurations/     # Manual analysis reports and configs

├── data/
│   ├── data_description.md
│   ├── processed_data/
│   └── raw_data/

├── docs/
│   └── Markdown documentation for analytic modules and clustering

├── scripts/
│   ├── analytic_scores_merging_execution.py
│   ├── archive_execution_scripts/
│   ├── artifact_production_execution/
│   ├── behavioural_volatility_feature_engineering.py/
│   ├── clustering/
│   ├── cross_domain_/
│   ├── interaction_mode_score/
│   └── revenue_score/

├── source_code_package/
│   ├── __init__.py
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/

└── tests/
    ├── __init__.py
    └── Unit tests for configuration, clustering, scoring, and feature engineering

## 5. Setup and Installation

In terms of how to gain the correct setup to be able to utilise this project's repository, the following must be attended to in terms of installing dependencies and gaining the right environment of which is reproducible. 

The first thing to consider is to prerequisites for utilisation of this repository. This only includes simple things such as cross-platform operating system, a Python version of 3.8 or higher (as specified in the pyproject.toml) and the use of github to clone the repository for use. 

Once this repository has been cloned, the installation method of simply using pyproject.toml to install the correct dependencies is recommended. A suggested way of doing so would be to run pip install -e in the terminal to read the configuration from pyproject.toml and automatically install all the dependencies listed there.

A good suggestion to ensure that all configurations and functionality througout the repository work as intended is to run the files in the tests folder also to hunt any bugs or issues. These test files are also built into the github actions continuous integration (CI).

Of course more linking to usage, but a strong way of checking that your setup is correct is to run the analytics-pipeline.yml workflow in github actions to see if this project's repository outputs based on the given configurations and data as it should. 

## 6. Usage

Relating more specifically to the usage of this repository, now that the setup and necessary installation for this project is explained, the following is an explanation of the intended usage of the repository's functionality to achieve the desired output and further use this repository with different data and configurations. Of course, the purpose of the output produced in this project is as a glimpse into the type of analytics that would be possible in extension of what is currently built. Therefore, the usage to be explained only produces a certain scope of opportunities even though the end scope in practical implementation of this type of project could be much larger. 

In terms of utilising the model data and configurations from which the manual analysis report is built, the way in which this is completed is by leaving the configurations the same as those shown in the Manual_Analysis_and_Configurations subfolder of the artifacts folder and using the currently held new_raw_data_polygon.csv file as the inputted raw data. This process then involves running the analytics-pipeline.yml workflow in the github actions section of the repository on the github user interface. Given successfull completion of this step, you can access and download the produced artifacts folder through the github actions interface at the artifacts section in the workflow run. This artifacts folder that can be downloaded produces the model dashboard which relates to the manual analysis report produced; a perfect show and explanation of the type of analytics being pursued with this project and how they relate to the business relevance of this project.

In regards to the other side of this project's functionality, being the ability for users to analyse alternative data (and in future iterations of this project in practical implementation, potentially automatically flowing data) and configurations, the only difference in the way that this process works is that a user can change configuations by making changes in the config subfolder of the Source_Code_Packages folder and by also adding new data files to the raw_data subfolder of the data folder (and respectively changing the raw data paths in the configurations). The key thing to remember here is that no automatically updating recommendations and analysis is produced when the dashboard changes as a result of updated data and configurations, therefore the premise of this "self-serve" analytics is very limited. This concept of "self-serve" analytics is slightly explored here in what is possible in actual implemetation of this type of a project because a user can perform their own analysis on different configurations and data by analysing a reproducible and ever-updating dashboard of visualisations and statistics held in the artifacts folder in the Dashboards subfolder.

## 7. Methods and Technical Details

The following explanations explore all the models, feature engineering and other processes related to the production of all methods in this repository for the production of analytic scores. Given the detailed nature of this explanation, the structure of this is in giving a brief description of what is being looked at in each of the analytic scores (and machine learning methods) followed by their actual derivation and related formulas and processes in what has been built. In each of the analytic scores that have been built throughout this repository, certain processes such as data preprocessing take the same or similar forms; therefore, these explanations will be included in one of the analytic score derivations and simply referred to in all other needed areas. 

Analytic scores and relevant models produced in this repository:

- **Behavioural Volatility Score**: 

The Behavioural Volatility Score quantifies how variable or erratic a wallet's transaction behavior is over time. It measures the inconsistency in transaction amounts and timing for each wallet. High scores indicate wallets with irregular, unpredictable activity, while low scores suggest regular, stable transaction patterns. The score is typically derived using statistical analysis of transaction histories, such as calculating the standard deviation or other measures of dispersion in transaction frequency and amounts. This helps identify users whose behavior deviates from the norm, which can be useful for segmentation, risk assessment, or anomaly detection. 

The behavioural volatility score takes on a core formula based on a composite of three components produced using formualas that integrate features from the raw data for each wallet. The formula for behavioural volatility score = √(0.35×Financial + 0.40×Activity + 0.25×Exploration) where "Financial", "Activity" and "Exploration" are each components with separate derivations. Financial volatility is produced as the coefficient of variation for transfer amounts: USD_TRANSFER_STDDEV / AVG_TRANSFER_USD. Activity volatility utilises all 10 of the event count columns in the raw data and is derived with the following formula: activity volatility = 0.40×CV + 0.30×Variance_Ratio + 0.30×Gini where CV is the std(event_counts) ÷ mean(event_counts), Variance_Ratio is the actual_variance ÷ expected_uniform_variance and the Gini takes the following formula (2×Σ(i×sorted_counts)) ÷ (n×Σ(counts)) - (n+1)/n. And finally, exploration volatility takes PROTOCOL_DIVERSITY, INTERACTION_DIVERSITY, TOKEN_DIVERSITY AND TX_PER_MONTH to in the following formula: √(mean(diversity_features) ÷ TX_PER_MONTH). The processing steps for this score are slightly different to that of the others, firstly, the raw components are calculated using the method described above. From here, normalisation is applied to each of the components where financial takes on robust scaling (median/IQR based), activity takes on z-score capping at plus or minus 3 sigma, and exploration taking on log and z-score transformation. From here of course, these components are summed with the described weights and this sum is then square rooted to produce a normalised score between 0 and 1. 

- **Revenue Proxy Score**: 

Estimates the potential revenue contribution of each wallet by combining several aspects of user activity. It is calculated as a weighted sum of three main components: transaction activity (frequency of transactions multiplied by their average value), DEX/DeFi engagement (the number of decentralized exchange and DeFi events, weighted by average transfer value), and bridge activity (the total volume transferred via bridges). By aggregating these factors, the score highlights wallets that are likely to be high-value users based on their transaction patterns and engagement with DeFi and cross-chain services. Of course, the key caveat to this analytic score is that in use of actual, valuable data this would be taken from a CEX's data and would be accurate. Therefore, this score is only produced to show how this key part of a dataset would be intended to be used in actuality.

The revenue proxy score takes on the following formula: 0.4×Transaction_Activity + 0.35×DeFi_Activity + 0.25×Bridge_Activity. Transaction activity is produced using AVG_TRANSFER_USD × TX_PER_MONTH. This slightly differs in complication to the DeFi activity (which also factors in DEX activity) using the formula: (DEX_EVENTS + DEFI_EVENTS) × AVG_TRANSFER_USD. And finally, the last component of bridge activity is simply the BRIDGE_TOTAL_VOLUME_USD from the raw data. The processing steps for this production is less complex than that of other scores; the output is a raw USD-denominated revenue proxy score.

- **Cross Domain Engagement Score**: 

The cross domain engagement score is a metric that quantifies how diversely a cryptocurrency wallet interacts across different event types or domains. It is calculated using shannon entropy, which measures the unpredictability or diversity of event participation for each wallet. A score near 1 means the wallet's activity is evenly spread across many event types (high engagement diversity). A score near 0 means the wallet's activity is concentrated in few or just one event type (low engagement diversity). A key issue with this score explored throughout is the fact that an overwhelming amount of wallets show a cross domain engagement of 0, not because they have low engagement, but because they have no engagement with the event categories in the dataset. This is extremely important to remember in interpretation and a potential flaw to improve upon in reflection, despite still being useful in interpretation of wallets with cross domain engagement scores above 0.

The shannon entropy used to calculate this score can be seen to be of the following formula: -Σ(pi × log₂(pi)) ÷ log₂(n), where pi is the proportion of each event type and n is the number of active event categories. The event categories included here involve all ten of the event types seen in the raw data. Before the calculation utilising the previously described formula, the zero proportions are filtered out, a natural logarithm of log base 2 is also of course used (seen in the formula). From here, the shannon entropy is normalised by dividing each wallets score from the previous formula by log₂(active_categories) in order to scale this to a 0 to 1 range to consider maximum possible entropy logic, maximum entropy occurs when activity is equally distributed across all categories.

- **Interaction Mode Score**: 

A median value is produced for each of DEX_EVENTS, CEX_EVENTS, BRIDGE_EVENTS and DEFI_EVENTS from clustering results where a cluster median for each feature is selected from the most prominant cluster for that event type. The distance score measures how much a wallet's behavior deviates from the median, after normalizing for feature variability and weighting by the wallet's activity profile. A higher score means the wallet acts more unusually compared to typical behavior, with the score reflecting both the size and importance of these differences. The typical behaviour being a value of an EVENT feature engagement that is prominant and high. A key issue with these scores is in production and complexity of interpretation, something that is reflected upon consistently; the score makes it more difficult to see whether a wallet's event value is 0 or more and doesn't add as much additional information about the wallet's position in comparison to the rest of the dataset as initially intended. However, this does not take away from the value of this score and is something used in reflection in later explanations.

The interaction mode scores are defined by first running the below UMAP + HDBSCAN clustering pipeline with 4 different target interaction features, being DEX_EVENTS, CEX_EVENTS, DEFI_EVENTS and BRIDGE_EVENTS (this clustering is completed on the full dataset, as well as the cluster datasets from the activity based clustering to provide larger scope, although not primarily used). From here, the clustering results are ran through the following cluster selection method: ranking by selection score which is equal to the non_zero_proportion × median_nonzero_value. This is where there is also a minimum activity threshold where 30% of wallets in a cluster must have non-zero values, the cluster size must be greater than 50 wallets (most likely already achieved in configurations) and if no clusters meet the threshold then the highest activity cluster is selected. From this ranking, the highest scoring cluster by selection score is selected for each of the 4 event types, and then from here the median of non-zero values is extracted from each of these clusters to be used as a "model" value for interaction with each event type. This then relates to individual wallet scoring, because in each wallet the raw distance is calculated from the median value produced for each event type to be seen as this interaction mode score. A number of normalised distance versions are also produced but are not used in the dashboards and manual analysis.

- **Activity-Based Clustering (UMAP + HDBSCAN)**: 

Groups wallets into clusters based on their activity patterns using a two-step pipeline: UMAP for dimensionality reduction and HDBSCAN for density-based clustering. This model reveals natural groupings or archetypes in the user base, helping to identify segments with similar behaviors. A key thing to note in analysing this score is to understand that the clusters are still rough in their accuracy; although it it true that distributions of data show differences in activity level of wallets, some wallets from one cluster may be more/less active than that of another cluster in a particular area.

The process for this UMAP + HDBSCAN clustering with the configurations aimed at activity-based clustering (of course these can be changed to target other behaviours) first starts with preprocessing of features before put into UMAP dimensionality reduction. The activity based features selected are TX_PER_MONTH, PROTOCOL_DIVERSITY, ACTIVE_DURATION_DAYS and TOTAL_TRANSFER_USD; these features for each wallet are log transformed (apart from TX_PER_MONTH and ACTIVE_DURATION_DAYS, due to the nature of what they represent) and then scaled by taking away the mean value then dividing this by the standard deviation. 

Moving into the UMAP dimensionality reduction, the model configurations target an output of 2 umap components and this process (Uniform Manifold Approximation and Projection) was used for dimensionality reduction, producing a 2D representation of the high-dimensional feature space. This allows similar wallets to be placed close together while preserving the overall structure of the data, making clusters and behavioral patterns easier to visualize and interpret.

Looking at the second component of this pipeline, the HDBSCAN (Hierarchical Density Based Spatial Clustering of Applications with Noise) clustering algorithm takes the dimensionality reduced data from UMAP, and applies this to produce cluster labels. Without diving into how this algorithm works, it assigns wallets to clusters based on density patterns while labeling outliers as noise, making it well-suited for uncovering behavioral archetypes in complex, high-dimensional data.

The above doesn't dive into the depths of how both of these algorithms work in themselves, but instead more how they are applied to this case.

## 8. Results and Insights

The main examples of how the results of this project can be used to gain insight into the data can be seen in the Manual_Analysis_and_Recommendations_Report.md file outputted in the artifacts folder. This file dives into great depth on how to explore the various distributions summarised and interpretable with statistics and visualisations in the dashboard.

This file makes use of "model" data and configurations to allow for a mock output that can be used to explain how to interpret this type of analytics. From here, it can be seen by users how you could go about inputting your own data or configurations to tailor analytics to what you want to look at from the perspective of a CEX stakeholder in actuality.

## 9. Key Learnings and Building Approach

A clear evaluation of the analytic scores produced in this project can be seen in the Manual_Analysis_and_Recommendations_Report.md, this dives into the flaws identified with the processes chosen for each of these analytic scores and also how they work together. In terms of what this section of the README looks at, this will look into the key flaws with the approach to building the analytic scores in terms of developing practices (code format and more) as well as a more rounded evaluation of the business relevance and application of what is being looked at through what has been built. The key improvements and areas to be worked on and considered in future projects are explained in the section below this. 

One of the main areas to be considered here is development of my own skills regarding repository development practices. This ties in with use of AI which is considered below, and essentially is learning what the standard, strong practices are with things such as code format, versioning control, documentation and making sure this repository is usable for others. In terms of code formatting, I took an approach of ensuring simplicity and reproducibility of code output over complex systems. I believe my approach here proved successful, the use of AI to produce small functions in each file with every piece of code understood and checked allowed for strong results. However, of course, I believe that in areas I could have improved in ensuring a common format is used across all files in terms of docstrings and general format which is something I look to improve upon in future projects. 

In addition to the above, I believe something that I have learnt as a result of getting to the end of this project is the importance of strong versioning control.

## 10. Next Steps Possible With Actual Application



## 11. Acknowledgements and References



## 12. Link to TLADS Canvases



## 13.  License



## 14. 

## Author

[Tom Davey](https://github.com/tomjedavey)

**READ THE BELOW FOR ADDITIONAL NOTES ON HOW THE TLADS CANVASES RELATE TO THE ACTUAL MODELS BUILT IN THE PROJECT TO PRODUCE THE ANALYTICAL OUTPUT**

Note on Metrics and Business KPIs:
This project uses only onchain public data. However, the TLADS canvases include references to business metrics and KPIs that would typically be available within a centralized exchange (e.g., revenue per user, session frequency). These are included to develop and demonstrate the business-aligned thinking behind the modeling, even though they are not computed or used directly in the final models.

# **ONE THING TO POTENTIALLY THINK ABOUT WHEN COMING TO THE END / PROJECT OUTPUT - IS THE REVENUE CONTRIBUTION SCORE USELESS IF A CEX CAN ACTUALLY TRACK REVENUE GAINED PER USER?? - MAYBE CHANGE THIS FUNCTIONALITY BACK TO SOMETHING PREDICTIVE USING TIME-SERIES DATA FROM SNOWFLAKE WHERE POSSIBLE. NEED TO EVALUATE THIS TO A LARGE EXTENT**
