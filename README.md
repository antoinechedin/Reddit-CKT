# Reddit Continuous Knowledge Tracing

This experience is divided in two parts: the prediction and the Knowledge Tracing (KT).

## Karma prediction
In this part, we try to predict the karma of a reddit post with different features. 
These features are:
* Length
* Polarity
* Spelling Error Ratio (SER)
* [Gunning-Fog Index](https://en.wikipedia.org/wiki/Gunning_fog_index) (FOG)
* [Flesch Reading Ease](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests) (FK)
* [Automated Readability Index](https://en.wikipedia.org/wiki/Automated_readability_index) (ARI)

Since Reddit posts have a title and a text, all features have to be computed on both.

Here we assume that we already have those metrics computed; for this example, our dataset has the following structure:

|id     |author              |created_utc  |karma |selftext_polarity|title_polarity|  ...  |title_ARI|
|-------|--------------------|------------:|-----:|----------------:|-------------:|:-----:|--------:|
|ert45d |CalmBeforeTheEclipse|1.510077495E9|   645|           0.5645|        0.1287|  ...  |        3|
|t52sf3 |CalmBeforeTheEclipse|1.486116616E9|   840|           0.2354|        0.0124|  ...  |        2|
|4frt23 |PasDePamplemousse   |1.505047838E9|     5|           0.7861|       0.05634|  ...  |        5|
|...    |...                 |          ...|   ...|              ...|           ...|  ...  |      ...|
|spkl24 |Palm32              |1.51235602E9 |     1|           0.8611|        0.6785|  ...  |        2|

The fields `id`, `author` and `created_utc` aren't necessary for the karma prediction but are required
for the KT.

An implementation of a linear regressor in Python is available in the directory `src/main/python/`. 

> The script uses [SKlearn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), 
> A `requirements.txt` file is located next to the script, with which you can easily install the dependencies using:
> ```
> pip install -r requirements.txt
> ```

> Some parameters may require changes before being able to use the script, such as the input folder, 
> output_folder, etc.  

## Knowledge Tracing

The second part is the actual KT. We want to perform two of them, one on the *real karma*, that will be 
used as ground truth for the other KT. That other KT is applied on the karma we preicted with our model during the first phase. 
That will allow us to compare them and to compute the error of the second KT. We use a Java app to apply the Knowledge Tracing.

### REDDIT CKT.JAR

This java app does the following:

* Execute a Python file. This file creates a model that tries to predict the Karma value. 
Exports the predicted Karma into a csv file.
    * This file can be edited to produce any prediction. It must however indicate at its 
    end that it has finished so that the Java app may continue, by printing `END OF SCRIPT`.
* Read the predictions made by the Python script.
* Reduce and center data: put all scores between 0 and 1 because KT needs bound values.
* Compute Karma prediction RMSE and MAE.
* For each threshold to test:
	* Execute a cross-validation to apply Knowledge Tracing on predicted Karma.
	* Apply Knowledge Tracing on real Karma.
	* Compute RMSE and MAE on those KTs.
* Export files with statistics on models:
	* params.csv with RMSEs, MAEs, and parameters obtained for KT.
	* usergraphs.json: Allows the drawing of a graph for each user comparing KT, Karma, 
	predicted Karma, predicted KT for an input threshold. Before exporting this json file, 
	the app asks the user for a threshold to make those graphs with. You can check the params.csv to choose an appropriate threshold.
	
The Java app takes as an argument a path to a properties file. This files contains all necessary
input data such as paths to files, different values for cross validations, etc.
Details on those arguments are contained as comments in the sample settings.properties file in this archive.


## Datasets
Datasets are not uploaded on the Git repository, but a sample one is available in the release.

### GENERAL
#### Main input:

* Datasets with postID, authorID, score, created_utc (in datasets directory)
* Number of folds in Cross validation
* Number of thresholds to test
* Maximum and minimum scores to bound input scores (optional)

#### Output:

* Compared performance of the Knowledge Tracing model applied on any prediciton model and
that prediction model.
* Parameters generated and used by the Knowledge Tracing model.
* Graphs to study the Knowledge of the studied data.

#### Release
The release archive contains the following:

* A RedditCKT.jar file, the main app to resolve our problem.
* A sample reddit.properties file, containing all paramters for the RedditCKT.jar app.
* A linear_reg.py file, a sample python script that produces a model to predict on the input data.
* A dataset-splitter.jar file (inside the datasets directory), an app that can split a dataset
into several subsets.
* A sample dataset "dataset-full.csv" in the "datasets" directory and its subsets in the subdirectory "5_fold_test".


### DATASET SPLITTER
`datasets` folder also contains a `dataset-splitter.jar` app that splits a dataset into several subsets. Arguments for this jar are:

```
<input_dataset_csv> <score_attribute_name> <subset_attribute_name> <output1_csv> [output2_csv] [output3_csv]...
```

(The number of subets is deduced by the number of output files specified.)
This splitter keeps problems of the same sequence together, and also keep distributions together. After splitting, it randomizes the problem order inside the subsets.
This split is used to make several datasets for cross-validaton.
With this jar, there is a sample splitter-5_folds_reddit.bat file, that can execute this app with working arguments.


## Sources
* Pre analyses were done on [Google Big Query](https://bigquery.cloud.google.com/dataset/fh-bigquery:reddit_posts)(in order to only retrieve the most relevant author and posts)
* Reddit datasets were gathered from http://files.pushshift.io/reddit/submissions/
