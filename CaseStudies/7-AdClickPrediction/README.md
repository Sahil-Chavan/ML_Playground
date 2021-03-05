# AdClickPrediction-CaseStudy

<h2>1.1 Problem Description </h2>
- Introduction: <br> Clickthrough rate (CTR)-
is a ratio showing how often people who see your ad end up clicking it. Clickthrough rate (CTR) can be used to gauge how well your keywords and ads are performing.
- CTR is the number of clicks that your ad receives divided by the number of times your ad is shown: clicks ÷ impressions = CTR. For example, if you had 5 clicks and 100 impressions, then your CTR would be 5%.
- Each of your ads and keywords have their own CTRs that you can see listed in your account.
- A high CTR is a good indication that users find your ads helpful and relevant. CTR also contributes to your keyword's expected CTR, which is a component of Ad Rank. Note that a good CTR is relative to what you're advertising and on which networks.
> Credits: Google (https://support.google.com/adwords/answer/2615875?hl=en) 
<p> Search advertising has been one of the major revenue sources of the Internet industry for years. A key technology behind search advertising is to predict the click-through rate (pCTR) of ads, as the economic model behind search advertising requires pCTR values to rank ads and to price clicks.<b> In this task, given the training instances derived from session logs of the Tencent proprietary search engine, soso.com, participants are expected to accurately predict the pCTR of ads in the testing instances. </b></p>

<h2>1.2 Source/Useful Links </h2>
__ Source __ : https://www.kaggle.com/c/kddcup2012-track2 <br>
__ Blog __ :https://hivemall.incubator.apache.org/userguide/regression/kddcup12tr2_dataset.html

<h2> 1.3 Real-world/Business Objectives and Constraints </h2>
Objective: Predict the pClick (probability of click) as accurately as possible.
Constraints: Low latency, Interpretability.

<h2>1.4 Mapping the Real-world to a Machine Learning problem </h2>
<h3>1.4.1 Type of Machine Learning Problem </h3>
It is a regression problem as we predicting CTR = #clicks/#impressions

<h3>1.4.2 Performance metric </h3>
Souce : https://www.kaggle.com/c/kddcup2012-track2#Evaluation <br>
- Hence I am going to use ROC
