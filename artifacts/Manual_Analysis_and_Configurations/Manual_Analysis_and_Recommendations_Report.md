### Manual Analysis and Recommendations Report of Dashboard Based Off Model Configurations and Data

## Objective and Explanation of This File and its Purpose:

This file is put in place to show a manual analysis and interpetation of the results of the pipelines built in this github repository in a way that will model the type of interpretation possible with the use of this repository. 

Of course, this project is a minimum viable version of what is possible in use of this data and in terms of what could be built with further CEX data available and in the context of actually putting this concept into use. Therefore, the output of this data science and analytics project is only a small part of what could be built. The ability of the stakeholder using this repository to input new data and utilise new configurations to gain updated analytics of a range of blockchains leads way to the possibility of a self-serve analytics making use of the models built. A feedback-based UEX is another possibility, utilising AI and the impact of stakeholders' opinions to constantly improve the value to key decision making available. More on these concepts is talked about in TLADs documentation along with the README; the main thing to note here is that this analysis report of the "model run" is simply to show how this analytics can be interpreted and what is possible for future developments and use of this repository and if it were to be actually utilised. 

Structure of this report:

1. Executive Summary
2. Business Initiative Context
3. Stakeholder Empathy
4. Modelling Business Entities Utilised
5. Mapping Analytic Score Analysis to Recommendations
6. Evaluation of further improvements possible

## 1. Executive Summary

The purpose of the production of these analytic scores and their interpretation in this report is to provide analytic value to a CEX through providing information related to the key decisions that affect a certain business initiative. The business initiative for a CEX being targeted here is to improve user retention and personalised engagement strategies for Centralised Crypto Exchanges (CEXs - user retention, churn prediction and behavioural segmentation) to increase fee revenue.

The way in which this aims to be acheived is through targeting the following use case to look at user segmentation based on wallet behavioural archetypes. This is likely to improve user retention and personalised engagement strategies as it allows a CEX stakeholder to identify a wallet by its types of behaviour and from there make sure that the strategies deployed on that user are best suited to the type of CEX user that they are, along with overall understanding of shifts in their user-base. Of course, in order to turn the functionality in this repository into use in this way, more CEX-specific data would be required to link wallets with users (proposing a number of considerations with privacy policies and more also), but this does not mean that the premise is not valuable to consider. 

In terms of the key findings of the output in this project with the displayed "model" configurations and data, it is clear to identify a number of "behavioural archetypes" of wallets where groups show similar data that relates to a type of behaviour that could be considered in key decision making. The archetypes being looked at in the report as examples include: "erratic speculators", "DeFi power users", and "omnichain explorers". Throughout the data and results displayed, there are a number of other possible groups to explored and a number of possible other analytic scores that could be built to do so, as is explained in later parts of this report. 

## 2. Business Initiative Context

When understanding and analysing the output of a data pipeline such as this repository's, it is vital to understand the business initiative being chased in order to take the right approach in interpreting the results of what has been produced. The business initiative in this case is to improve user retention and personalised engagement strategies for Centralised Crypto Exchanges (CEXs - user retention, churn prediction and behavioural segmentation) to increase fee revenue. 

User retention in this case refers to keeping a user utilising the services of a CEX and this is aimed to be improved through being looking at the nanoeconomics (individual behaviours and propensities) of the data and informing CEX stakeholders on potential strategies that target specific behaviours seen in groups of wallets/users in order to avoid churn and keep these users utilising that CEX. This of course then drives a more strong user base and increased revenues and growth of a CEX when this analytic value is utilised properly. 

When looking at KPIs that would be targeted to be improved given this business initiative and that could be tracked with the correct types of data, key areas include: churn rate (%) for all the users of a CEX, CEX retention by archetype, withdrawal-deposit ratio, reactivation attribution and more. This is explored in great depth throughout the TLADS canvases which essentially consider how to go about building a project with these types of goals by thinking in a actual business applicable way.


## 3. Stakeholder Empathy 


The analytic value produced and possible through interpretation of the output of this repository is designed with the premise of informing a wide range of CEX stakeholders (possible in actuality through improvements in data integration). Therefore, the following touches upon a number of these stakeholders and maps how the output of this project relates more specifically to each. This is explored in a pre-production stage in the TLADS canvases in this repository that shows a more broad view than this explanation which focuses on areas sepecific to the small portions of analytics possible that have been produced in the results being looked at. 

One of the most important stakeholders to consider given this use case is the head of user retention strategies. The key decisions that the results of this project aid of course is in this stakeholder deciding which users to deploy certain strategies on. An example of what this could look like given what has been produced is to potentially target "erratic speculators" (which are likely high churn risk) and curate watchlists to more volatile coins or deploy other similar strategies to take advantage of their typical propensities. This is explored in much greater depth in the actual recommendations and more later in this report. 

Additionally, another vital stakeholder to consider is those working in product management at a CEX. The way in which these analytics can assist these stakeholders is through informing on the types of groups (or "behavioural archetypes") with which to personalise the user interface and services of the CEX to. For example, it may be best for this team to build greater user interface to explore new DeFi protocols to target the "DeFi power users" group. 

And finally, another stakeholder to touch upon is the marketing and user engagement team. The way in which stakeholders in this team likely use the analytics possible with this project is slightly different to that of other stakeholders in that there is a greater opportunity to analyse temporal shifts in the overall behaviour of a set of wallets/users to know how to change marketing being used. To explain this better, if a CEX marketing team was to understand that the group of "omnichain explorers" was growing in the users seen in the CEX over time, then this may reflect a change in the overall crypto markets where more of this demographic of wallets and CEX users are likely to use a wider range of protocols, coins and applications. This could therefore drive the outbound marketing strategies to be more tailored to this type of behaviour seen in users. 

Of course, many more stakeholders would be impacted by the full integration of this project and the fulfilment of its potential, but the above is a great demonstration of the type of stuff that is possible as it stands. More is explored in the TLADS canvases (Thinking Like a Data Scientist, a method explained by Bill Schmarzo).


## 4. Modelling Businesses and Scores


The following is a brief explanation of the analytic scores (and an additional model produced) used in nanoeconomics of wallet data to identify groups that relate to certain "behavioural archetypes":

- **Behavioural Volatility Score**: The Behavioural Volatility Score quantifies how variable or erratic a wallet's transaction behavior is over time. It measures the inconsistency in transaction amounts and timing for each wallet. High scores indicate wallets with irregular, unpredictable activity, while low scores suggest regular, stable transaction patterns. The score is typically derived using statistical analysis of transaction histories, such as calculating the standard deviation or other measures of dispersion in transaction frequency and amounts. This helps identify users whose behavior deviates from the norm, which can be useful for segmentation, risk assessment, or anomaly detection. 

- **Revenue Proxy Score**: estimates the potential revenue contribution of each wallet by combining several aspects of user activity. It is calculated as a weighted sum of three main components: transaction activity (frequency of transactions multiplied by their average value), DEX/DeFi engagement (the number of decentralized exchange and DeFi events, weighted by average transfer value), and bridge activity (the total volume transferred via bridges). By aggregating these factors, the score highlights wallets that are likely to be high-value users based on their transaction patterns and engagement with DeFi and cross-chain services.

- **Cross Domain Engagement Score**: The cross domain engagement score is a metric that quantifies how diversely a cryptocurrency wallet interacts across different event types or domains. It is calculated using Shannon entropy, which measures the unpredictability or diversity of event participation for each wallet. A score near 1 means the wallet's activity is evenly spread across many event types (high engagement diversity). A score near 0 means the wallet's activity is concentrated in few or just one event type (low engagement diversity)

- **Interaction Mode Score**: A median value is produced for each of DEX_EVENTS, CEX_EVENTS, BRIDGE_EVENTS and DEFI_EVENTS from clustering results where a cluster median for each feature is selected from the most prominant cluster for that event type. The distance score measures how much a wallet's behavior deviates from the median, after normalizing for feature variability and weighting by the wallet's activity profile. A higher score means the wallet acts more unusually compared to typical behavior, with the score reflecting both the size and importance of these differences. The typical behaviour being a value of an EVENT feature engagement that is prominant and high.

- **Activity-Based Clustering (UMAP + HDBSCAN)**: Groups wallets into clusters based on their activity patterns using a two-step pipeline: UMAP for dimensionality reduction and HDBSCAN for density-based clustering. This model reveals natural groupings or archetypes in the user base, helping to identify segments with similar behaviors.

The above analytic scores are built in order to turn the raw data into more interpretable scores to be used by those looking at the analytic output produced. Making use of these scores, a key way of interpreting and utilising what is outputted is in identifying groups of wallets that show common propensities in each of the scores and informing key decision making with this. 


## 5. Mapping Analytic Score Recommendations to Analysis


The following takes on the process for which this report document is mainly defined to achieve; analysing the output of the pipeline with the "model" configurations and data in order to give clear examples of how the output of this project can be interpreted to use the described use case in achieving the described business intiative. The actual data that this is modelled on is described in data_description.md in the data folder of this repository, and the model configurations for this pipeline run being analysed are saved in this subfolder of the artifacts folder (Manual_Analysis_Report_and_Configurations).

The way in which this is structured is by looking at the overall data's analytic score distributions, followed by the three "behavioural archetypes" for which visualisations are produced in the dashboard, and from there linking to the analytic scores and clustering output. Of course there is a great deal of additional analysis and interpretations that can be gained by looking at the visualisations and statistics seen in the dashboard, but for the purpose of understanding the ways in which the output of this project can be interpreted only the "behavioural archetype" visualisations are analysed below:

- Overall data distributions: 
Looking at the initial analytic score distribution visualisations and their statistics, it is clear to see that a great deal show extreme variance (every dashboard visualisation is of the 5th to 95th percentile to accomodate this). The behavioural volatility score distribution shows a more normal distribution around a mean of 0.58, with a smaller peak at lower values. The cross domain engagement score shows the large majority of wallets to have a value close to 0, with rising proportions of wallets (from a count of wallets near zero just after the proportion around 0) as the cross domain score increases. The revenue score proxy shows a greater deal of the distribution at lower scales, with decreasing proportions as the revenue score proxy increases and extreme outliers at high values of revenue score proxy. And finally, across the interaction mode scores, all show a large majority around the distance values that represent a events value of 0, with median values for DEX_EVENTS, CEX_EVENTS, BRIDGE_EVENTS and DEFI_EVENTS of 29, 2, 29 and 12. However, value can be seen in terms of identifying groups by looking at the proportions of wallets that display values below these median values which produce a interaction mode less than these values.

- UMAP dimensionality reduction + HDBSCAN clustering results:
Taking a more in depth look at the results of this pipeline, the intended purpose of the model configurations was to be able to find a rough split based on activity. This therefore clusters wallets into two clusters (-1 being the produced noise) with fairly even populations (cluster 0 with 8713 and cluster 1 with 11390, 0.4% of wallets being noise). I won't explain the depths of how the clustering metrics work, but with the understanding that importance is seen in comparison between different runs, it can be said to take this clustering with a degree of skeptisism as although differences are shown in analytic score distributions per cluster, there is a large degree of variation. This does not invalidate the impact of these findings however, 

- Erratic Speculators Visualisations.