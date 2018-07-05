package fr.diblois.ckt;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Properties;

import fr.diblois.ckt.data.KTParametersStats;
import fr.diblois.ckt.data.KTTIResults;
import fr.diblois.ckt.util.Problem;
import fr.diblois.ckt.util.Sequence;
import fr.diblois.ckt.util.Utils;

public class RedditCKT
{

	public static KTParametersStats avg_parameters, stdev_parameters;
	public static final ArrayList<Sequence> dataset = new ArrayList<>();
	public static String dataset_directory, results_directory;
	public static int folds, kttis;
	public static double karma_rmse, kt_rmse, kt_rmse_stdev;
	private static final ArrayList<String> log = new ArrayList<>();
	public static boolean minFixed, maxFixed;
	public static double minValue, maxValue;
	public static final Properties settings = new Properties();
	public static final ArrayList<Sequence> testset = new ArrayList<>();

	private static void exportParams()
	{
		String[][] data = new String[11][2];
		data[0][0] = "P(L0) Avg";
		data[0][1] = Utils.toString(avg_parameters.startKnowledge);

		data[1][0] = "P(L0) Stdev";
		data[1][1] = Utils.toString(stdev_parameters.startKnowledge);

		data[2][0] = "P(T) Avg";
		data[2][1] = Utils.toString(avg_parameters.transition);

		data[3][0] = "P(T) Stdev";
		data[3][1] = Utils.toString(stdev_parameters.transition);

		data[4][0] = "P(G) Avg";
		data[4][1] = Utils.toString(avg_parameters.guess);

		data[5][0] = "P(G) Stdev";
		data[5][1] = Utils.toString(stdev_parameters.guess);

		data[6][0] = "P(S) Avg";
		data[6][1] = Utils.toString(avg_parameters.slip);

		data[7][0] = "P(S) Stdev";
		data[7][1] = Utils.toString(stdev_parameters.slip);

		data[8][0] = "Karma RMSE";
		data[8][1] = Utils.toString(karma_rmse);

		data[9][0] = "KT RMSE Avg";
		data[9][1] = Utils.toString(kt_rmse);

		data[10][0] = "KT RMSE Stdev";
		data[10][1] = Utils.toString(kt_rmse_stdev);
	}

	private static void exportUserGraphs()
	{
		// TODO Auto-generated method stub

	}

	private static void finish()
	{
		exportParams();
		exportUserGraphs();

		try
		{
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File("log.txt")));
			for (String message : log)
				writer.write(message + "\n");
			writer.close();
		} catch (IOException e)
		{
			log("File log.txt couldn't be opened: " + e.getMessage());
			e.printStackTrace();
		}
	}

	/** Executes a single fold iteration in the cross validation. */
	private static KTTIResults fold(int fold)
	{
		testset.clear();
		// Execute Python script

		return null;
	}

	public static Problem getProblem(String id)
	{
		for (Sequence s : dataset)
			for (Problem p : s.problems)
				if (p.name.equals(id)) return p;
		return null;
	}

	public static Sequence getSequence(String id)
	{
		for (Sequence s : dataset)
			if (s.name.equals(id)) return s;
		return null;
	}

	/** Prints a message and adds it to the log. */
	public static void log(String message)
	{
		log.add(message);
		System.out.println(message);
	}

	public static void main(String[] args)
	{
		if (args.length >= 1 && args[0].equals("help"))
		{
			System.out.println("Usage: [settings_file_path]");
			return;
		}

		log("Starting Reddit CKT.");

		FileUtils.readSettings(args.length == 0 ? "settings.properties" : args[0]);
		FileUtils.readGroundTruthAndMetrics();
		// Execute DNNr.py
		FileUtils.readPredictions(RedditCKT.results_directory + File.separator + "dnn_predictions.csv");
		karma_rmse = Stats.computeKarmaRMSE();

		// Iterate for cross validation
		KTTIResults[] crossValidation = new KTTIResults[folds];
		for (int fold = 0; fold < folds; ++fold)
			crossValidation[fold] = fold(fold);

		finish();
	}

}
