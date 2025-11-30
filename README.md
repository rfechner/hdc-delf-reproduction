# hdc-delf-reproduction

An attempt to reproduce the results table in [Hyperdimensional computing as a framework for systematic aggregation of image descriptors](https://openaccess.thecvf.com/content/CVPR2021/papers/Neubert_Hyperdimensional_Computing_as_a_Framework_for_Systematic_Aggregation_of_Image_CVPR_2021_paper.pdf). 

I've collected results for the OxfordRobotCar dataset and evaluated where ground truth is available. Main experiment run script is `run_exp.py`.

I've noticed large variance in the results when changing random seeds, which is why I'm reporting the mean and median average precision over 10 seeds (as the approximate integral of the Recall-Precision Curve) rounded to nearest cent.

|Database|Query|Mean|Median|Min|Max|Reported in paper|
|------|------|------|------|---|---|------|
|2014-12-09|2015-05-19|0.90|0.91|0.86|0.92|0.91|
|2014-12-09|2015-08-28|0.74|0.75|0.69|0.81|0.71|


