# hdc-delf-reproduction

An attempt to reproduce the results table in [Hyperdimensional computing as a framework for systematic aggregation of image descriptors](https://openaccess.thecvf.com/content/CVPR2021/papers/Neubert_Hyperdimensional_Computing_as_a_Framework_for_Systematic_Aggregation_of_Image_CVPR_2021_paper.pdf). 

I've collected results for the OxfordRobotCar dataset and evaluated where ground truth is available. Main experiment run script is `run_exp.py`.

I've noticed large variance in the results when changing random seeds, which is why I'm reporting the mean and median average precision over 10 seeds (as the approximate integral of the Recall-Precision Curve) rounded to nearest cent.

|Database|Query|Mean|Median|Min|Max|Reported in paper|
|------|------|------|------|---|---|------|
|2014-12-09|2015-05-19|0.89|0.89|0.86|0.93|0.91|
|2014-12-09|2015-08-28|0.74|0.74|0.69|0.79|0.71|
|2014-12-09|2014-11-25|0.87|0.88|0.81|0.90|0.82|
|2014-12-09|2014-12-16|0.51|0.50|0.39|0.67|0.80|
|2015-05-19|2015-02-03|0.80|0.82|0.69|0.91|0.78|
|2015-08-28|2014-11-25|0.74|0.74|0.70|0.77|0.71|

#### Normalizing Descriptors over sample and feature dimensions
Instead of normalizing samples of shape `[nsamples, nfeatures, din]` per-image i.e. over axis 1, we're normalizing over
axes (0, 1).

|Database|Query|Mean|Median|Min|Max|Reported in paper|
|------|------|------|------|---|---|------|
|2014-12-09|2015-05-19|0.86|0.86|0.83|0.89|0.91|
|2014-12-09|2015-08-28|0.72|0.71|0.68|0.76|0.71|
|2014-12-09|2014-11-25|0.87|0.87|0.85|0.89|0.82|
|2014-12-09|2014-12-16|0.55|0.56|0.47|0.69|0.80|
|2015-05-19|2015-02-03|0.77|0.76|0.73|0.85|0.78|
|2015-08-28|2014-11-25|0.71|0.71|0.68|0.74|0.71|

Interestingly, this drops average precision over all database/query setups *except* 2014-12-09--2014-12-16, which this implementation struggled with especially.





