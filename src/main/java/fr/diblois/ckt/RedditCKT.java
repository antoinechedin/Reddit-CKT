package fr.diblois.ckt;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Properties;
import java.util.Random;
import java.util.Scanner;

import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;
import com.eclipsesource.json.PrettyPrint;

import fr.diblois.ckt.data.Gaussian;
import fr.diblois.ckt.data.KTFIResults;
import fr.diblois.ckt.data.KTParameters;
import fr.diblois.ckt.data.KTParametersStats;
import fr.diblois.ckt.data.KTTIResults;
import fr.diblois.ckt.util.Problem;
import fr.diblois.ckt.util.PythonLogger;
import fr.diblois.ckt.util.Sequence;
import fr.diblois.ckt.util.Utils;

public class RedditCKT
{

	public static KTParametersStats avg_parameters, stdev_parameters;
	public static String column_order, column_problem, column_sequence, column_score, column_karma_predicted;
	public static final ArrayList<Sequence> dataset = new ArrayList<>();
	public static String dataset_directory, results_directory, predictions_file, script;
	public static int folds, kttis;
	public static double karma_rmse, karma_mae;
	private static final ArrayList<String> log = new ArrayList<>();
	public static boolean minFixed, maxFixed;
	public static double minValue, maxValue;
	public static Process process;
	public static boolean processFinished = false;
	public static final Properties settings = new Properties();
	public static final ArrayList<ArrayList<Sequence>> subsets = new ArrayList<>();

	/** Uses the input <code>threshold</code> to determine the correctness of the problem. */
	private static void applyThreshold(double threshold)
	{
		for (Sequence sequence : dataset)
			for (Problem problem : sequence.problems)
			{
				problem.isCorrect = problem.prediction >= threshold;
				problem.isTruthCorrect = problem.karma >= threshold;
			}
	}

	/** Determines P(L0), P(T), P(G), P(S). Analyzes the {@link Main#learningSet learning set} and returns the parameters.
	 *
	 * @param trainset
	 * @param onGroundTruth */
	private static KTParameters computeParameters(ArrayList<Sequence> trainset, boolean onGroundTruth)
	{
		double kStart = 0, mTransition = 0, mGuess = 0, mSlip = 0;

		// P(L0)
		int count = 0;
		for (Sequence sequence : trainset)
		{
			kStart += (onGroundTruth ? sequence.knowledgeSequenceGT.get(0) : sequence.knowledgeSequence.get(0));
			++count;
			/* Tried using more than one for starting knowledge, but had close to no impact. if (sequence.knowledgeSequence.size() > 1) { kStart += sequence.knowledgeSequence.get(1); ++s; } if (sequence.knowledgeSequence.size() > 2) { kStart += sequence.knowledgeSequence.get(2); ++s; } */
		}
		kStart /= count;

		// mu(P(T)), mu(P(G)), mu(P(S))
		int tCount = 0, gCount = 0, sCount = 0;
		for (Sequence sequence : trainset)
		{
			KTParameters params = sequence.computeProbabilities(kStart, onGroundTruth);
			if (!Double.isNaN(params.transition))
			{
				mTransition += params.transition;
				++tCount;
			}
			if (!Double.isNaN(params.guess.mean))
			{
				mGuess += params.guess.next();
				++gCount;
			}
			if (!Double.isNaN(params.slip.mean))
			{
				mSlip += params.slip.next();
				++sCount;
			}
		}
		mTransition /= tCount;
		mGuess /= gCount;
		mSlip /= sCount;

		// sigma(P(G)), sigma(P(S))
		// double sTransition = 0;
		double sGuess = 0, sSlip = 0;
		for (Sequence sequence : trainset)
		{
			KTParameters params = onGroundTruth ? sequence.parametersGT : sequence.parameters;
			// if (!Double.isNaN(params.transition)) sTransition += Math.pow(params.transition - mTransition, 2);
			if (!Double.isNaN(params.guess.mean)) sGuess += Math.pow(params.guess.mean - mGuess, 2);
			if (!Double.isNaN(params.slip.mean)) sSlip += Math.pow(params.slip.mean - mSlip, 2);
		}
		// sTransition = Math.sqrt(sTransition / tSize);
		sGuess = Math.sqrt(sGuess / gCount);
		sSlip = Math.sqrt(sSlip / sCount);

		// Can happen if threshold is too low or too high
		if (Double.isNaN(mGuess)) mGuess = 0;
		if (Double.isNaN(sGuess)) sGuess = 0;
		if (Double.isNaN(mSlip)) mSlip = 0;
		if (Double.isNaN(sSlip)) sSlip = 0;

		return new KTParameters(kStart, mTransition, new Gaussian(mGuess, sGuess), new Gaussian(mSlip, sSlip));
	}

	public static void executePython()
	{
		RedditCKT.log("Executing Python script. Logs will be printed when script ends.");
		String command = "python " + script + " \"" + dataset_directory + "\" \"" + results_directory + "\"";
		try
		{
			new Thread(new PythonLogger()).start();
			process = Runtime.getRuntime().exec(command);

			while (!processFinished)
			{
				if (!process.isAlive())
				{
					System.err.println("Python script ended abruptly! Error: ");
					InputStream s = process.getErrorStream();
					byte[] bytes = new byte[s.available()];
					s.read(bytes);
					System.err.println(new String(bytes, StandardCharsets.UTF_8));
					System.exit(0);
				}
				Thread.sleep(100);
			}
			System.out.println("Python script finished.");
		} catch (final Exception e)
		{
			log("Error while executing python script!");
			e.printStackTrace();
		}
	}

	private static void exportParams(KTTIResults[] thresholds)
	{
		String[] header = { "Threshold", "mae_pred", "mae_kt", "rmse_pred", "rmse_kt", "corrects_karma", "corrects_pred", "PL0_avg_pred", "PL0_stdev_pred",
				"PT_avg_pred", "PT_stdev_pred", "PS_avg_pred", "PS_stdev_pred", "PG_avg_pred", "PG_stdev_pred", "PL0", "PL0 stdev", "PT", "PT stdev", "PS",
				"PS stdev", "PG", "PG stdev" };
		String[][] data = new String[thresholds.length + 1][header.length];
		data[0] = header;

		for (int t = 1; t <= thresholds.length; ++t)
		{
			KTTIResults threshold = thresholds[t - 1];
			if (threshold == null) continue;
			data[t][0] = Utils.toString(threshold.threshold * (maxValue - minValue) + minValue);
			data[t][1] = Utils.toString(karma_mae);
			data[t][2] = Utils.toString(threshold.ktMAE());
			data[t][3] = Utils.toString(karma_rmse);
			data[t][4] = Utils.toString(threshold.ktRMSE());
			data[t][5] = Utils.toString(threshold.correctsGT());
			data[t][6] = Utils.toString(threshold.corrects());
			data[t][7] = Utils.toString(threshold.pl0_avg());
			data[t][8] = Utils.toString(threshold.pl0_stdev());
			data[t][9] = Utils.toString(threshold.pt_avg());
			data[t][10] = Utils.toString(threshold.pt_stdev());
			data[t][11] = Utils.toString(threshold.ps_avg());
			data[t][12] = Utils.toString(threshold.ps_stdev());
			data[t][13] = Utils.toString(threshold.pg_avg());
			data[t][14] = Utils.toString(threshold.pg_stdev());
			data[t][15] = Utils.toString(threshold.pl0_gt());
			data[t][16] = Utils.toString(threshold.pl0_gtstdev());
			data[t][17] = Utils.toString(threshold.pt_gt());
			data[t][18] = Utils.toString(threshold.pt_gtstdev());
			data[t][19] = Utils.toString(threshold.ps_gt());
			data[t][20] = Utils.toString(threshold.ps_gtstdev());
			data[t][21] = Utils.toString(threshold.pg_gt());
			data[t][22] = Utils.toString(threshold.pg_gtstdev());
		}

		FileUtils.exportData(new File(results_directory + File.separator + "params.csv"), data);
	}

	private static void exportUserGraphs(KTTIResults[] thresholds)
	{
		double t = 0;
		Scanner sc = new Scanner(System.in);
		System.out.println("Type in threshold to make usergraphs with (unreduced).");
		t = sc.nextDouble();
		sc.close();

		log("Executing best threshold again to gather user graphs.");
		threshold((t - minValue) / (maxValue - minValue));// Don't reduce threshold because it's already stocked as reduced
		JsonArray root = Json.array();
		for (Sequence sequence : dataset)
		{
			JsonObject seq = Json.object();
			seq.add("name", sequence.name);

			JsonArray karma = Json.array(), karma_pred = Json.array(), kt = Json.array(), kt_pred = Json.array();
			for (Problem problem : sequence.problems)
			{
				karma.add(Json.value(Double.isFinite(problem.karma) ? problem.karma : Double.MAX_VALUE));
				karma_pred.add(Json.value(Double.isFinite(problem.prediction) ? problem.prediction : Double.MAX_VALUE));
				kt.add(Json.value(Double.isFinite(problem.expectedKnowledge.mean) ? problem.expectedKnowledge.mean : Double.MAX_VALUE));
				kt_pred.add(Json.value(Double.isFinite(problem.knowledge.mean) ? problem.knowledge.mean : Double.MAX_VALUE));
			}

			seq.add("karma", karma);
			seq.add("karma_pred", karma_pred);
			seq.add("kt", kt);
			seq.add("kt_pred", kt_pred);
			root.add(seq);
		}

		try
		{
			String data = root.toString(PrettyPrint.indentWithTabs());
			BufferedWriter bw = new BufferedWriter(new FileWriter(new File(results_directory + File.separator + "usergraphs.json")));
			bw.write(data);
			bw.close();
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	/** Determines which Problems should be used to compute Precision. */
	private static void findRepresentativeProblems(ArrayList<Sequence> set)
	{
		for (Sequence sequence : set)
		{
			if (settings.getProperty("smooth_rmse").equals("true"))
			{
				Problem best = sequence.finalProblem();
				// Find problem with smallest distance to expected
				for (Problem p : sequence.problems)
					if (Math.abs(p.expectedKnowledge.mean - p.knowledge.mean) < Math.abs(best.expectedKnowledge.mean - best.knowledge.mean)) best = p;

				// Use it and all problems after
				for (int i = sequence.problems.indexOf(best); i < sequence.problems.size(); ++i)
					sequence.problems.get(i).isRepresentative = true;
			} else sequence.finalProblem().isRepresentative = true;
		}
	}

	private static void finish(KTTIResults[] thresholds)
	{
		exportParams(thresholds);
		exportUserGraphs(thresholds);

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

	/** Executes a single fold iteration in the cross validation. */
	private static KTFIResults knowledgeTracing(int fold, double threshold, ArrayList<Sequence> trainset, ArrayList<Sequence> testset)
	{
		Utils.random = new Random(1);
		// log("Executing Knowledge Tracing...");
		for (Sequence sequence : trainset)
		{
			sequence.findKnowledgeSequence(false);
			sequence.findKnowledgeSequence(true);
		}

		double corrects = 0, correctsGT = 0, total = 0;
		for (Sequence sequence : trainset)
			for (Problem problem : sequence.problems)
			{
				if (problem.isCorrect) ++corrects;
				if (problem.isTruthCorrect) ++correctsGT;
				++total;
			}

		corrects = corrects * 100 / total;
		correctsGT = correctsGT * 100 / total;

		KTParameters params = computeParameters(trainset, false);
		KTParameters paramsGT = computeParameters(trainset, true);
		for (Sequence sequence : testset)
		{
			sequence.computeKnowledge(params, false);
			sequence.computeKnowledge(paramsGT, true);
		}
		findRepresentativeProblems(testset);

		return new KTFIResults(fold, paramsGT, params, correctsGT, corrects, Stats.computeKTRMSE(testset), Stats.computeMAE(testset));
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
		if (script != null) executePython();
		FileUtils.readPredictions(RedditCKT.results_directory + File.separator + RedditCKT.predictions_file);
		reduceAndCenter();
		karma_rmse = Stats.computeKarmaRMSE();
		karma_mae = Stats.computeKarmaMAE();

		double thresholdIncrement = (maxValue - minValue) * 1. / kttis;

		// Iterate for cross validation
		KTTIResults[] thresholds = new KTTIResults[kttis + 1];
		int i = 0;
		// System.out.println(thresholdIncrement);
		for (double threshold = minValue; threshold <= maxValue; threshold += thresholdIncrement)
			thresholds[i++] = threshold((threshold - minValue) / (maxValue - minValue));

		RedditCKT.log("Exporting data.");
		finish(thresholds);
		log("Finished!");
	}

	private static void reduceAndCenter()
	{
		for (Sequence sequence : dataset)
			for (Problem problem : sequence.problems)
			{
				if (!minFixed && problem.karma < minValue) minValue = problem.karma;
				if (!minFixed && problem.prediction < minValue) minValue = problem.prediction;
				if (!maxFixed && problem.karma > maxValue) maxValue = problem.karma;
				if (!maxFixed && problem.prediction > maxValue) maxValue = problem.prediction;
			}
		for (Sequence sequence : dataset)
			for (Problem problem : sequence.problems)
			{
				problem.karma = (problem.karma - minValue) / (maxValue - minValue);
				problem.prediction = (problem.prediction - minValue) / (maxValue - minValue);

				if (problem.karma > 1) problem.karma = 1;
				if (problem.karma < 0) problem.karma = 0;
				if (problem.prediction > 1) problem.prediction = 1;
				if (problem.prediction < 0) problem.prediction = 0;
			}
	}

	private static KTTIResults threshold(double threshold)
	{
		RedditCKT.log("Executing knowledge tracing with threshold: " + threshold);
		applyThreshold(threshold);

		ArrayList<Sequence> trainset = new ArrayList<>(), testset = new ArrayList<>();
		KTFIResults[] validation = new KTFIResults[folds];
		for (int fold = 0; fold < folds; ++fold)
		{
			trainset.clear();
			testset.clear();

			testset.addAll(subsets.get(fold));
			trainset.addAll(dataset);
			trainset.removeAll(testset);

			validation[fold] = knowledgeTracing(fold, threshold, trainset, testset);
		}
		return new KTTIResults(threshold, validation);
	}

}
