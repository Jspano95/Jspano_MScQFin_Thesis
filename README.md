# Jspano_MScQFin_Thesis
Supporting code for UvA Masters of Quantitative Finance thesis in CDS/Bonds arbitrage trading

The main file which contains the entire stylized notebook is: Jspano_MSc_QuantFin_Public_Github_V2.00
This notebook has been designed to condense months of research into an easily digestable process that outlines all the steps taken for my own thesis work, as well as the additional research work done under Dr. T. Ladika, regarding arbitrage trading / fixed income markets research. 

The .py file or the notebook contain the single firm, quick-results compiler designed to facilitate a quick and easy understanding of the process that focuses on a single firm (rather than using groupby and by sector etc.) 

## THESIS EXECUTIVE SUMMARY
* According to financial theory, both CDS and bonds are linked by their relationship in measuring credit risk. Moreover, the theory of no-arbitrage pricing would sugest that both markets price credit risk equally- as reflected by their prices- otherwise, opportunities risk-free arbitrage profits would arise. In practice, there are a multitude of factors which lead to market frictions that can prevent this so-called 'equilibrium' or no-arbitrage state from persisting.

* The existing body of research, supported by underlying economic theory would suggests that this equilirium state of the two asset prices should be cointegrated. This idea will be tested by utilising a sample size far greater than any previous paper; furthermore, the sample size will be long enough to incorporate both the global financial crisis and the euro-zone crisis. Thus, the dynamics of the CDS-bond relationship can be empirically tested over time.

* This empirical testing will be done by examining if a cointegrating relationship exist between the CDS and BOND for a particular firm. If the presence of this cointegrating relationship is confirmed, a vector error correction model (VECM) will be applied and subsequent price-discovery speed mechanisms (introduced in the relevant section) calculated to determine the underlying dynamics of each series.

  * Where no cointegration is found, the series can be used to perform a vector autoregression (VAR) and test the causality through a Granger causality test; however, this mechanism is unable to conclude the price discovery or adjustment mechanisms, but will provide an outline of causality through the predictive power of (p) lags on the current price.
