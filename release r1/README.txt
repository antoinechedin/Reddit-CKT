GENERAL ------------------------------
Main input:
- datasets with postID, authorID, score, created_utc (in datasets directory)
- Number of folds in Cross validation
- Number of thresholds to test
- Maximum & minimum scores to bound input scores (optional)

Output:
- Compared performance of the Knowledge Tracing model applied on any prediciton model and that prediction model.
- Parameters generated and used by the Knowledge Tracing model.
- Graphs to study the Knowledge of the studied data.


This archive contains the following:
- A RedditCKT.jar file, the main app to resolve our problem.
- A sample reddit.properties file, containing all paramters for the RedditCKT.jar app.
- A linear_reg.py file, a sample python script that produces a model to predict on the input data.
- A dataset-splitter.jar file (inside the datasets folder), an app that can split a dataset into several subsets.

REDDIT CKT.JAR -----------------------
This java app does the following:

- Execute a Python file. This file creates a model that tries to predict the Karma value. Exports the predicted Karma into a csv file.
	* This file can be edited to produce any prediction. It must however indicate at its end that it has finished so that the Java app may continue, by printing "END OF SCRIPT".

- Read the predictions made by the Python script.

- Reduce & center data: put all scores between 0 and 1 because KT needs bound values.

- Compute Karma prediction RMSE & MAE.
- For each threshold to test:
	* Execute a cross-validation to apply Knowledge Tracing on predicted Karma.
	* Apply Knowledge Tracing on real Karma.
	* Compute RMSE & MAE on those KTs.

- Export files with statistics on models:
	* params.csv with RMSEs, MAEs, and parameters obtained for KT.
	* usergraphs.json: Allows the drawing of a graph for each user comparing KT, Karma, predicted Karma, predicted KT for an input threshold. 
	Before exporting this json file, the app asks the user for a threshold to make those graphs with. You can check the params.csv to choose an appropriate threshold.
	
Java app takes as an argument a path to a properties file. This files contains all necessary input data such as paths to files, different values for cross validations, etc.
Details on those arguments are contained as comments in the sample settings.properties file in this archive.


DATASET SPLITTER -------------------------
"datasets" folder also contains a "dataset-splitter.jar" app that splits a dataset into several subsets. Arguments for this jar are:
<input_dataset_csv> <score_attribute_name> <subset_attribute_name> <output1_csv> [output2_csv] [output3_csv]...
(The number of subets is deduced by the number of output files specified.)
This splitter keeps problems of the same sequence together, and also keep distributions together. After splitting, it randomizes the problem order inside the subsets.
This split is used to make several datasets for cross-validaton.
With this jar, there is a sample splitter-5_folds_reddit.bat file, that can execute this app with working arguments.
