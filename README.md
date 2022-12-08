# Bayesian Learning via Stochastic Gradient Langevin Dynamics by Max Welling and Yee Whye Teh

**Contributors** : Amal Chaoui and Narjisse Lasri

This github repository presents the source code of the work performed  the article ["Bayesian Learning via Stochastic Gradient Langevin Dynamics"](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) (SGLD) by Max Welling and Yee Whye Teh.

It comprises the following files:
 - `BML_project_report.pdf` : Project report explaining the contents of the paper, emphazing its strong and weak points, and showing the results of the application of SGLD to real data as well as a contribution in form of a variant sampling method based on HMC with batch gradient updates.
 - `sgld.py` : Presents the main functions used to implement SGLD. 
 - `sgld_toy.ipynb` : Application of SGLD to toy synthetic data. 
 - `sghd.py` : Presents the main functions used to implement our proposed sampling method: Stochastic Gradient Hamiltonian Dynamics (SGHD). 
 - `sghd_toy.ipynb` : Application of our contribution (Stochastic Gradient Hamiltonian Dynamics) to toy synthetic data. The notebook describes the implementation and the results. 
 - `sgld_sghd_real_dataset.ipynb` : Application of SGLD and SGHD to real data (The Bank Marketing Dataset). 
 - `bank.csv` : [The Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset) (BDM) csv file consisting of 16 features collected from 11162 customers by a Portuguese banking institution. The classification goal is to predict whether the client will subscribe to a term deposit after a phone calls marketing campaign.
 
**Dependencies** : To run these files, it is necessary to download the following libraries : `numpy`, `matplotlib`, `sklearn`, `scipy`, `pymc3`. 

**Some results** : 

![My Image](image/plot.png)
<p style="text-align: center;">Figure 1 : Decision boundaries obtained by SGLD and SGHD for different classification tasks.</p>
<br /><br />


![My Image](image/SGLD.png)
<p style="text-align: center;">Figure 2 : Trace and auto-correlation plots of some coefficients of θ obtained using SGLD on the  real dataset (BDM).</p>
<br /><br />


![My Image](image/SGHD.png)
<p style="text-align: center;">Figure 3 : Trace and auto-correlation plots of the parameters of θ obtained using SGHD on the real dataset (BDM). </p>
<br /><br />




