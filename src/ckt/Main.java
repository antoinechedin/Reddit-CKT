package ckt;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Properties;
import java.util.Random;
import java.util.Scanner;
import java.util.function.Predicate;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import ckt.KTParameters.Gaussian;

public class Main
{
	/** The Precision for aggregated metrics. */
	static double aggregatedPrecision;
	/** Constant to add to the metrics aggregation. */
	static double aggregationBase;
	/** Stores all the Sequences used for cross validation. */
	static ArrayList<Sequence> allSequences;
	/** The parameters for main Knowledge. */
	static KTParameters[] expectedParameters;
	/** When iterating over different thresholds, the value of the current one. -1 if no iteration. */
	static int iterationThreshold = -1;
	/** Stores the Sequences used as learning set. */
	static ArrayList<Sequence> learningSet;
	static ArrayList<String> log = new ArrayList<String>();
	/** The parameters for main Knowledge. */
	static KTParameters[] mainParameters;
	static boolean maxSet = false, minSet = false;
	/** The available metrics. */
	static ArrayList<Metric> metrics;
	static HashMap<Metric, KTParameters[]> metricsKTParams; // In the context of the aggregated KT, indicates for each metric the parameters of the sub-skill KT corresponding to this metric
	static double minScore, maxScore;
	/** path to output / input directory **/
	final static String PATH = "";// "D:\\Workspace\\KnowledgeTracing\\";
	/** Settings from settings.properties */
	static Properties settings;
	/** Stores the Sequences used as testing set. */
	static ArrayList<Sequence> testingSet;
	/** The number of Sequences to use as testing set for each step in cross validation. */
	static int testingSize;
	/** Correctness threshold. */
	static double threshold;
	/** The total number of problems. */
	static int totalProblems;
	/** The number of steps in cross validation. */
	static int validations;

	/** Aggregates Knowledge found for each metric.
	 * 
	 * @return True if succeeded. */
	private static boolean aggregateMetrics()
	{
		log("Aggregating metrics...");
		if (settings.getProperty("aggregation_type").equals("svm") || settings.getProperty("aggregation_type").equals("input"))
		{
			// Applying SVM script
			exportData(new File(PATH + "resultat.csv"), outputSVM(settings.getProperty("aggregation_type").equals("svm")));

			if (settings.getProperty("aggregation_type").equals("svm"))
			{
				if (!new File(settings.getProperty("aggregation_value")).exists())
				{
					System.out.println("SVM file not found");
					return false;
				}
				// running external Python SVM

				Process p = null;
				String command = "python " + settings.getProperty("aggregation_value");
				try
				{
					p = Runtime.getRuntime().exec(command);
				} catch (final IOException e)
				{
					e.printStackTrace();
				}

				// Wait to get exit value
				try
				{
					p.waitFor();
					final int exitValue = p.waitFor();
					if (exitValue == 0) System.out.println("Successfully executed the command: " + command);
					else
					{
						System.out.println("Failed to execute the following command: " + command + " due to the following error(s):");
						try (final BufferedReader b = new BufferedReader(new InputStreamReader(p.getErrorStream())))
						{
							String line;
							if ((line = b.readLine()) != null) System.out.println(line);
						} catch (final IOException e)
						{
							e.printStackTrace();
						}
					}
				} catch (InterruptedException e)
				{
					e.printStackTrace();
				}

				/* try { System.out.println("Running python SVM..."); Process p = Runtime.getRuntime().exec("python " + settings.getProperty("aggregation_value"));
				 * 
				 * while (p.isAlive()) Thread.sleep(100); } catch (IOException e) { e.printStackTrace(); return false; } catch (InterruptedException e) { e.printStackTrace(); }
				 * 
				 * System.out.println("SVM completed."); */
			} else
			{
				System.out.println("Waiting... Press Enter when the weights file is ready.");
				new Scanner(System.in).nextLine();
			}

			// Gathering SVM data
			String weightsData = Utils
					.readTextFile(settings.getProperty("aggregation_type").equals("svm") ? PATH + "weights.txt" : settings.getProperty("aggregation_value"));
			weightsData = weightsData.replaceAll("\\n", "").replaceAll("\\[", "").replaceAll("\\]", "");
			while (weightsData.contains("  "))
				weightsData = weightsData.replaceAll("  ", " ");
			String[] weights = weightsData.split(" ");
			if (weights.length < metrics.size()) log("SVM failed !");
			for (int i = 1; i <= metrics.size(); ++i)
				metrics.get(i - 1).weight = Double.parseDouble(weights[i]);// * 1. / 1000000.;

			// Aggregating
			double[] aggregated = new double[Sequence.DRAWS];
			for (Sequence sequence : allSequences)
				for (Problem problem : sequence.problems)
				{
					for (int d = 0; d < Sequence.DRAWS; ++d)
					{
						aggregated[d] = 0;
						for (Metric metric : metrics)
							aggregated[d] += metric.weight * problem.metricKnowledge.get(metric).next();
					}
					problem.aggregatedKnowledge = Utils.makeGaussian(aggregated);
				}
		} else
		{
			ScriptEngine script = new ScriptEngineManager().getEngineByName("nashorn");
			File file = null;
			if (settings.getProperty("aggregation_type").equals("script")) file = new File(settings.getProperty("aggregation_value"));

			int invalid = 0;
			double[] scores = new double[metrics.size()];
			double[] aggregated = new double[Sequence.DRAWS];
			for (Sequence sequence : allSequences)
				for (Problem problem : sequence.problems)
				{
					for (int d = 0; d < Sequence.DRAWS; ++d)
					{
						if (file == null)
						{
							aggregated[d] = aggregationBase;
							// if (allSequences.indexOf(sequence) <= 10 && d == 0) log();
							for (Metric metric : metrics)
							{
								/* if (allSequences.indexOf(sequence) <= 10 && d == 0) log(metric.name + ": " + metric.weight + " * " + metric.initialDistribution.unreduce(sequence.finalProblem().metricKnowledge.get(metric).next())); */
								aggregated[d] += metric.weight * metric.initialDistribution.unreduce(problem.metricKnowledge.get(metric).next());
							}
						} else try
						{
							for (int m = 0; m < metrics.size(); ++m)
								scores[m] = metrics.get(m).initialDistribution.unreduce(problem.metricKnowledge.get(metrics.get(m)).next());

							// Input: double[] containing scores for each metric for a single problem.
							// Output: aggregated score as a double.
							script.put("metrics", scores);
							script.eval(new BufferedReader(new FileReader(file)));
							aggregated[d] = (double) script.get("aggregated");
						} catch (FileNotFoundException e)
						{
							log("Couldn't find aggregation script: " + settings.getProperty("aggregation_script"));
							return false;
						} catch (Exception e)
						{
							log("Error while aggregating metrics:\n" + e.getMessage());
							return false;
						}
						if (aggregated[d] >= 1) aggregated[d] = 1;
					}
					problem.aggregatedKnowledge = Utils.makeGaussian(aggregated);
					if (problem.aggregatedKnowledge.mean > 1) ++invalid;

				}
			if (invalid != 0) log("Found " + invalid + " invalid knowledge values!");
		}

		// Center new metric values

		double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
		for (Sequence sequence : allSequences)
			for (Problem problem : sequence.problems)
			{
				double value = problem.aggregatedKnowledge.mean;
				if (value < min) min = value;
				if (value > max) max = value;
			}

		max -= min;
		for (Sequence sequence : allSequences)
			for (Problem problem : sequence.problems)
			{
				Gaussian value = problem.aggregatedKnowledge;
				problem.aggregatedKnowledge = new Gaussian((value.mean - min) / max, value.variation);
			}

		return true;

	}

	private static void applyKnowledgeTracing(Metric metric)
	{
		Utils.random = new Random(1);
		if (metric == null) log("Executing Knowledge Tracing...");
		else log("Executing Knowledge Tracing on metric \"" + metric.name + "\"...");
		if (metric != null) applyThreshold(metric);
		findKnowledgeSequences(metric);

		double corrects = 0, starts = 0, total = 0;
		for (Sequence sequence : allSequences)
		{
			if (metric == null ? sequence.problems.get(0).isCorrect : sequence.problems.get(0).isCorrectTemp) ++starts;
			for (Problem problem : sequence.problems)
			{
				if (metric == null ? problem.isCorrect : problem.isCorrectTemp) ++corrects;
				++total;
			}
		}

		starts = starts * 100 / allSequences.size();
		corrects = corrects * 100 / total;
		System.out.println("Correct starts: " + Utils.toString(starts) + "%");
		System.out.println("Correct problems: " + Utils.toString(corrects) + "%");

		KTParameters[] params = new KTParameters[validations + 2];
		if (metric == null) mainParameters = params;

		for (int i = 0; i < validations; ++i)
		{
			learningSet.clear();
			testingSet.clear();
			for (int j = 0; j < allSequences.size(); ++j)
				if (j < i * testingSize || j >= (i + 1) * testingSize) learningSet.add(allSequences.get(j));
				else testingSet.add(allSequences.get(j));

			params[i] = computeParameters(metric);
			computeKnowledge(params[i], metric);
		}
		if (metric == null) computeStats(params);
		else
		{
			// add-on here to track local KT Parameters for each metric in the context of aggregated KT
			if (metricsKTParams == null) metricsKTParams = new HashMap<Metric, KTParameters[]>();
			metricsKTParams.put(metric, params);
		}
	}

	/** Uses the input <code>threshold</code> to determine the correctness of the problem. */
	private static void applyThreshold(double threshold)
	{
		for (Sequence sequence : allSequences)
			for (Problem problem : sequence.problems)
				problem.isCorrect = problem.score >= threshold;
	}

	/** Uses the metric's <code>threshold</code> to determine the correctness of the problem. */
	private static void applyThreshold(Metric metric)
	{
		for (Sequence sequence : allSequences)
			for (Problem problem : sequence.problems)
				problem.isCorrectTemp = (!metric.thresholdReversed && problem.metricScores.get(metric) >= metric.threshold)
						|| (metric.thresholdReversed && problem.metricScores.get(metric) < metric.threshold);
	}

	/** 1) Removes empty sequences.<br />
	 * 2) Sorts problems in sequences chronologically.<br />
	 * 3) Reduces and centers problem scores between 0 and 1 if <code>center</code> is true.<br />
	 * 4) Reduces and centers metric scores between 0 and 1 if <code>centerMetrics</code> is true. */
	static void cleanSequences(boolean center, boolean centerMetrics)
	{
		allSequences.removeIf(new Predicate<Sequence>() {
			@Override
			public boolean test(Sequence t)
			{
				if (t.problems.size() == 0) log("Sequence " + t.name + " has no problems !");
				return t.problems.size() == 0;
			}
		});
		allSequences.sort(Comparator.naturalOrder());

		totalProblems = 0;
		for (Sequence sequence : allSequences)
			totalProblems += sequence.problems.size();

		if (center)
		{
			for (Sequence sequence : allSequences)
				for (Problem problem : sequence.problems)
					if (!minSet && problem.score < minScore) minScore = problem.score;
					else if (!maxSet && problem.score > maxScore) maxScore = problem.score;
			for (Sequence sequence : allSequences)
				for (Problem problem : sequence.problems)
					if (!minSet && problem.groundTruth < minScore) minScore = problem.groundTruth;
					else if (!maxSet && problem.groundTruth > maxScore) maxScore = problem.groundTruth;

			for (Sequence sequence : allSequences)
				for (Problem problem : sequence.problems)
				{
					problem.score = (problem.score - minScore) / (maxScore - minScore);
					if (problem.score > 1) problem.score = 1;
					if (problem.score < 0) problem.score = 0;

					problem.groundTruth = (problem.groundTruth - minScore) / (maxScore - minScore);
					if (problem.groundTruth > 1) problem.groundTruth = 1;
					if (problem.groundTruth < 0) problem.groundTruth = 0;
				}

			threshold = (threshold - minScore) / (maxScore - minScore);
		}

		if (metrics.size() != 0 && centerMetrics)
		{
			for (Metric metric : metrics)
			{
				double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
				for (Sequence sequence : allSequences)
					for (Problem problem : sequence.problems)
					{
						double score = problem.metricScores.get(metric);
						if (score < min) min = score;
						else if (score > max) max = score;
					}

				for (Sequence sequence : allSequences)
					for (Problem problem : sequence.problems)
						problem.metricScores.put(metric, (problem.metricScores.get(metric) - min) / (max - min));

				metric.threshold = (metric.threshold - min) / (max - min);
				metric.initialDistribution = new Gaussian(min, max - min);
			}
		}
	}

	/** @return The average L(n) value of the last problem in each of the input sequences based on the real helpfulness score not the ones outputed by one of our model. Warning: each P(Ln) is a Gaussian, this function only returns the average of Gaussian means */
	private static double computeAverageExpectedLearning(ArrayList<Sequence> sequences)
	{
		double sumLearning = 0; // contains the sum of all P(Ln) for the last problem of the input sequences
		for (Sequence sequence : sequences)
		{
			sumLearning += sequence.finalProblem().expectedKnowledge;
		}
		return sumLearning / sequences.size();
	}

	/** @param aggregated - If true, will compute the average of the last L(n) value for each sequence of problems for the aggregated Knowledge.
	 * @return The average L(n) value of the last problem in each of the input sequences. Warning: each P(Ln) is a Gaussian, this function only returns the average of Gaussian means */
	private static double computeAverageLearning(ArrayList<Sequence> sequences, boolean aggregated)
	{
		double sumLearning = 0; // contains the sum of all P(Ln) for the last problem of the input sequences
		for (Sequence sequence : sequences)
		{
			sumLearning += (aggregated) ? sequence.finalProblem().aggregatedKnowledge.mean : sequence.finalProblem().knowledge.mean;
		}
		return sumLearning / sequences.size();
	}

	private static double computeDeviationExpectedLearning(ArrayList<Sequence> sequences, double average)
	{
		double sumLearning = 0; // contains the sum of all deviations from P(Ln) to the average of P(Ln) for the last problem of the input sequences
		for (Sequence sequence : sequences)
		{
			sumLearning += Math.pow(sequence.finalProblem().expectedKnowledge - average, 2);
		}
		return Math.sqrt(sumLearning / sequences.size());
	}

	/** Add on by Nicolas for the DOLAP paper **/

	private static double computeDeviationLearning(ArrayList<Sequence> sequences, boolean aggregated, double average)
	{
		double sumLearning = 0; // contains the sum of all deviations from P(Ln) to the average of P(Ln) for the last problem of the input sequences
		for (Sequence sequence : sequences)
		{
			sumLearning += (aggregated) ? Math.pow(sequence.finalProblem().aggregatedKnowledge.mean - average, 2)
					: Math.pow(sequence.finalProblem().knowledge.mean - average, 2);
		}
		return Math.sqrt(sumLearning / sequences.size());
	}

	/** @param aggregated - If true, will compute the Precision for the aggregated Knowledge.
	 * @return The Precision for the Knowledge for the input sequences. */
	private static double computeInitialPrecision(ArrayList<Sequence> sequences)
	{
		double precision = 0;
		int problems;// Number of representative problems in current sequence
		for (Sequence sequence : sequences)
		{
			problems = -1;
			for (Problem problem : sequence.problems)
				if (problem.isRepresentative)
				{
					if (problems == -1) problems = sequence.problems.size() - sequence.problems.indexOf(problem);

					if (threshold == -1) precision += Math.pow(sequence.finalProblem().groundTruth - problem.score, 2) / problems;
					else precision += Math.pow(sequence.finalProblem().groundTruth - (problem.score >= threshold ? 1 : 0), 2) / problems;
				}
		}

		return Math.sqrt(precision / sequences.size());
	}

	/** @return The Precision for the Knowledge for the input sequences. */
	private static double computeInitialRMSE(ArrayList<Sequence> sequences)
	{
		double precision = 0;
		int problems;// Number of representative problems in current sequence
		for (Sequence sequence : sequences)
		{
			problems = -1;
			for (Problem problem : sequence.problems)
				if (problem.isRepresentative)
				{
					if (problems == -1) problems = sequence.problems.size() - sequence.problems.indexOf(problem);

					precision += Math.pow(sequence.finalProblem().groundTruth - problem.score, 2) / problems;
				}
		}

		return Math.sqrt(precision / sequences.size());
	}

	/** @param aggregated - If true, will compute the variation for the aggregated Knowledge.
	 * @return The variation for the final Knowledge for the input sequences. */
	private static double computeInitialVariation(ArrayList<Sequence> sequences)
	{
		double variation = 0;
		int count = 0, problems = -1;
		for (Sequence sequence : sequences)
		{
			problems = -1;
			++count;
			for (Problem problem : sequence.problems)
				if (problem.isRepresentative)
				{
					if (problems == -1) problems = sequence.problems.size() - sequence.problems.indexOf(problem);
					variation += sequence.finalProblem().score / problems;
				}
		}

		return variation / count;
	}

	/** Computes the Knowledge of the Sequences. */
	private static void computeKnowledge(KTParameters parameters, Metric metric)
	{
		for (Sequence sequence : testingSet)
			sequence.computeKnowledge(parameters, metric);
	}

	/** Determines P(L0), P(T), P(G), P(S). Analyzes the {@link Main#learningSet learning set} and returns the parameters.
	 * 
	 * @param metric */
	private static KTParameters computeParameters(Metric metric)
	{
		double kStart = 0, mTransition = 0, mGuess = 0, mSlip = 0;

		// P(L0)
		int count = 0;
		for (Sequence sequence : learningSet)
		{
			kStart += sequence.knowledgeSequence.get(0);
			++count;
			/* Tried using more than one for starting knowledge, but had close to no impact. if (sequence.knowledgeSequence.size() > 1) { kStart += sequence.knowledgeSequence.get(1); ++s; } if (sequence.knowledgeSequence.size() > 2) { kStart += sequence.knowledgeSequence.get(2); ++s; } */
		}
		kStart /= count;

		// mu(P(T)), mu(P(G)), mu(P(S))
		int tCount = 0, gCount = 0, sCount = 0;
		for (Sequence sequence : learningSet)
		{
			sequence.computeProbabilities(kStart);
			if (!Double.isNaN(sequence.parameters.transition))
			{
				mTransition += sequence.parameters.transition;
				++tCount;
			}
			if (!Double.isNaN(sequence.parameters.guess.mean))
			{
				mGuess += sequence.parameters.guess.next();
				++gCount;
			}
			if (!Double.isNaN(sequence.parameters.slip.mean))
			{
				mSlip += sequence.parameters.slip.next();
				++sCount;
			}
		}
		mTransition /= tCount;
		mGuess /= gCount;
		mSlip /= sCount;

		// sigma(P(G)), sigma(P(S))
		// double sTransition = 0;
		double sGuess = 0, sSlip = 0;
		for (Sequence sequence : learningSet)
		{
			// if (!Double.isNaN(sequence.parameters.transition)) sTransition += Math.pow(sequence.parameters.transition - mTransition, 2);
			if (!Double.isNaN(sequence.parameters.guess.mean)) sGuess += Math.pow(sequence.parameters.guess.mean - mGuess, 2);
			if (!Double.isNaN(sequence.parameters.slip.mean)) sSlip += Math.pow(sequence.parameters.slip.mean - mSlip, 2);
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

	/** @param aggregated - If true, will compute the Precision for the aggregated Knowledge.
	 * @return The Precision for the Knowledge for the input sequences. */
	private static double computePrecision(ArrayList<Sequence> sequences, boolean aggregated)
	{
		double precision = 0;
		int problems;// Number of representative problems in current sequence
		for (Sequence sequence : sequences)
		{
			problems = -1;
			for (Problem problem : sequence.problems)
				if (problem.isRepresentative)
				{
					if (problems == -1) problems = sequence.problems.size() - sequence.problems.indexOf(problem);
					if (aggregated && problem.aggregatedKnowledge == null) continue;

					if (threshold == -1)
					{
						precision += Math.pow(sequence.finalProblem().expectedKnowledge - (aggregated ? problem.aggregatedKnowledge : problem.knowledge).mean,
								2) / problems;
						/* if (Math.pow(sequence.finalProblem().expectedKnowledge - (aggregated ? problem.aggregatedKnowledge : problem.knowledge).mean, 2) / problems > 1) { System.out.println((aggregated ? problem.aggregatedKnowledge : problem.knowledge).mean); System.out.println(
						 * Math.pow(sequence.finalProblem().expectedKnowledge - (aggregated ? problem.aggregatedKnowledge : problem.knowledge).mean, 2) / problems); } */
					} else precision += Math.pow(sequence.finalProblem().expectedKnowledge
							- ((aggregated ? problem.aggregatedKnowledge : problem.knowledge).mean >= threshold ? 1 : 0), 2) / problems;
				}
		}

		return Math.sqrt(precision / sequences.size());
	}

	/** @param aggregated - If true, will compute the Precision for the aggregated Knowledge.
	 * @return The Precision for the Knowledge for the input sequences. */
	private static double computeRMSE(ArrayList<Sequence> sequences, boolean aggregated)
	{
		double precision = 0;
		int problems;// Number of representative problems in current sequence
		for (Sequence sequence : sequences)
		{
			problems = -1;
			for (Problem problem : sequence.problems)
				if (problem.isRepresentative)
				{
					if (problems == -1) problems = sequence.problems.size() - sequence.problems.indexOf(problem);
					if (aggregated && problem.aggregatedKnowledge == null) continue;

					precision += Math.pow(sequence.finalProblem().expectedKnowledge - (aggregated ? problem.aggregatedKnowledge : problem.knowledge).mean, 2)
							/ problems;
				}
		}

		return Math.sqrt(precision / sequences.size());
	}

	/** Uses the metrics to calculate the score of each problem.
	 * 
	 * @return True if it succeeded. */
	private static boolean computeScores()
	{
		log("Calculating scores...");
		ScriptEngine script = new ScriptEngineManager().getEngineByName("nashorn");
		File file = null;
		if (settings.getProperty("aggregation_type").equals("script")) file = new File(settings.getProperty("aggregation_value"));

		double[] scores = new double[metrics.size()];
		for (Sequence sequence : allSequences)
		{
			for (Problem problem : sequence.problems)
			{
				if (file == null)
				{
					problem.score = aggregationBase;
					// if (allSequences.indexOf(sequence) <= 10 && d == 0) log();
					for (Metric metric : metrics)
						problem.score += metric.weight * problem.metricScores.get(metric);

				} else try
				{
					for (int m = 0; m < metrics.size(); ++m)
						scores[m] = problem.metricScores.get(metrics.get(m));

					// Input: double[] containing scores for each metric for a single problem.
					// Output: aggregated score as a double.
					script.put("metrics", scores);
					script.eval(new BufferedReader(new FileReader(file)));
					problem.score = (double) script.get("aggregated");
				} catch (FileNotFoundException e)
				{
					log("Couldn't find aggregation script: " + settings.getProperty("aggregation_script"));
					return false;
				} catch (Exception e)
				{
					log("Error while aggregating metrics:\n" + e.getMessage());
					return false;
				}
			}

		}

		return true;
	}

	/** Computes statistics for the parameters. */
	private static void computeStats(KTParameters[] parametersArray)
	{
		// Mean
		double start = 0, transition = 0, guess = 0, slip = 0;
		for (int i = 0; i < parametersArray.length - 2; ++i)
		{
			start += parametersArray[i].startKnowledge;
			transition += parametersArray[i].transition;
			guess += parametersArray[i].guess.mean;
			slip += parametersArray[i].slip.mean;
		}

		// EDIT: NICO change here parametersArray.length by parametersArray.length - 2
		start /= (parametersArray.length - 2);
		transition /= (parametersArray.length - 2);
		guess /= (parametersArray.length - 2);
		slip /= (parametersArray.length - 2);

		// Variation
		double sStart = 0, sTransition = 0, sGuess = 0, sSlip = 0;

		// EDIT: NICO change here for sGuess and sSlip that are now considered as mean of their Gaussian distributions
		// I don not take into account the variation around the mean here to compute the standard deviation

		for (int i = 0; i < parametersArray.length - 2; ++i)
		{
			sStart += Math.pow(parametersArray[i].startKnowledge - start, 2);
			sTransition += Math.pow(parametersArray[i].transition - transition, 2);
			sGuess += Math.pow(parametersArray[i].guess.mean - guess, 2);
			sSlip += Math.pow(parametersArray[i].slip.mean - slip, 2);
		}

		sStart = Math.sqrt(sStart / (parametersArray.length - 2));
		sTransition = Math.sqrt(sTransition / (parametersArray.length - 2));
		sGuess = Math.sqrt(sGuess / (parametersArray.length - 2));
		sSlip = Math.sqrt(sSlip / (parametersArray.length - 2));

		parametersArray[validations] = new KTParameters(start, transition, new Gaussian(guess, sGuess), new Gaussian(slip, sSlip));
		parametersArray[validations + 1] = new KTParameters(sStart, sTransition, null, null);

		/* Old version more complex proposed by Clement. Maybe we should stick with it?
		 * 
		 * for (int i = 0; i < parametersArray.length - 2; ++i) { sStart += Math.pow(parametersArray[i].startKnowledge - start, 2); sTransition += Math.pow(parametersArray[i].transition - transition, 2); sGuess += Math.pow(parametersArray[i].guess.mean, 2) +
		 * Math.pow(parametersArray[i].guess.variation, 2); sSlip += Math.pow(parametersArray[i].slip.mean, 2) + Math.pow(parametersArray[i].slip.variation, 2); } sStart = Math.sqrt(sStart / parametersArray.length); sTransition = Math.sqrt(sTransition / parametersArray.length); sGuess =
		 * Math.sqrt(sGuess / parametersArray.length - Math.pow(guess, 2)); sSlip = Math.sqrt(sSlip / parametersArray.length - Math.pow(slip, 2));
		 * 
		 * parametersArray[validations] = new KTParameters(start, transition, new Gaussian(guess, sGuess), new Gaussian(slip, sSlip)); parametersArray[validations + 1] = new KTParameters(sStart, sTransition, null, null); */
	}

	/** @param aggregated - If true, will compute the variation for the aggregated Knowledge.
	 * @return The variation for the final Knowledge for the input sequences. */
	private static double computeVariation(ArrayList<Sequence> sequences, boolean aggregated)
	{
		double variation = 0;
		int count = 0, problems = -1;
		for (Sequence sequence : sequences)
		{
			problems = -1;
			++count;
			for (Problem problem : sequence.problems)
				if (problem.isRepresentative)
				{
					if (problems == -1) problems = sequence.problems.size() - sequence.problems.indexOf(problem);
					if (aggregated && problem.aggregatedKnowledge == null) continue;
					variation += (aggregated ? sequence.finalProblem().aggregatedKnowledge : sequence.finalProblem().knowledge).variation / problems;
				}
		}

		return variation / count;
	}

	private static boolean createMetrics(String inputMetrics, String inputWeight, String inputThreshold)
	{
		metrics = new ArrayList<Metric>();
		try
		{
			inputMetrics = inputMetrics.substring(1, inputMetrics.length() - 1).replaceAll(" ", "");
			if (inputMetrics.equals("")) return true;
			String[] m = inputMetrics.split(",");
			for (String metric : m)
				metrics.add(new Metric(metric));

			if (inputWeight != null)
			{
				inputWeight = inputWeight.substring(1, inputWeight.length() - 1).replaceAll(" ", "");
				if (inputWeight.equals("")) return true;
				String[] w = inputWeight.split(",");
				for (int i = 0; i < w.length - 1; ++i)
					metrics.get(i).weight = Utils.parseDouble(w[i]);
				aggregationBase = Utils.parseDouble(w[w.length - 1]);
			}

			if (inputThreshold != null)
			{
				if (!inputThreshold.startsWith("["))
				{
					double t = Double.parseDouble(inputThreshold);
					for (Metric metric : metrics)
					{
						metric.threshold = t;
						metric.thresholdReversed = metric.weight < 0;
					}
				} else
				{
					inputThreshold = inputThreshold.substring(1, inputThreshold.length() - 1).replaceAll(" ", "");
					if (inputThreshold.equals("")) return true;
					String[] t = inputThreshold.split(",");
					for (int i = 0; i < t.length && i < metrics.size(); ++i)
					{
						if (t[i].startsWith("<")) metrics.get(i).thresholdReversed = true;
						if (t[i].startsWith("<") || t[i].startsWith(">")) t[i] = t[i].substring(1);
						metrics.get(i).threshold = Utils.parseDouble(t[i]);
					}
				}
			}

			return true;
		} catch (Exception e)
		{
			log("Error while reading metrics: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
	}

	/** Reads the input files and creates the corresponding Sequences.
	 * 
	 * @param set - The set to store the Sequences in.
	 * @param input - The input file with the data.
	 * @return true if it succeeded. */
	private static boolean createSequences(ArrayList<Sequence> set, File input)
	{
		set.clear();
		log("Reading input file: " + input.getName());

		int sequenceID = -1, problemID = -1, expectedKnowledge = -1, order = -1, correctness = -1, score = -1;
		HashMap<Metric, Integer> metricIndex = new HashMap<Metric, Integer>();
		try
		{
			CSVParser parser = CSVParser.parse(input, Charset.defaultCharset(), CSVFormat.DEFAULT);

			int lines = 0;
			for (CSVRecord record : parser)
				if (parser.getCurrentLineNumber() == 1)
				{
					for (int i = 0; i < record.size(); ++i)
					{
						if (record.get(i).equals("sequence")) sequenceID = i;
						else if (record.get(i).equals("problem")) problemID = i;
						else if (record.get(i).equals("expected_knowledge")) expectedKnowledge = i;
						else if (record.get(i).equals("order")) order = i;
						else if (record.get(i).equals("correctness")) correctness = i;
						else if (record.get(i).equals("score")) score = i;
						for (Metric metric : metrics)
							if (record.get(i).equals(metric.name)) metricIndex.put(metric, i);

						if (expectedKnowledge == -1) expectedKnowledge = score;
					}
					for (Metric metric : metrics)
						if (!metricIndex.containsKey(metric))
						{
							System.out.println("Metric couldn't be found in input file: " + metric.name);
							return false;
						}
				} else try
				{
					Sequence sequence = findSequence(set, record.get(sequenceID));
					Problem p = order == -1 ? new Problem(record.get(problemID))
							: new Problem(record.get(problemID), Double.valueOf(record.get(order)).intValue());
					p.groundTruth = Utils.parseDouble(record.get(expectedKnowledge));
					p.isCorrect = correctness == -1 ? false : record.get(correctness).equals("1");
					p.score = score == -1 ? 0 : Utils.parseDouble(record.get(score));

					for (Metric metric : metrics)
						p.metricScores.put(metric, Utils.parseDouble(record.get(metricIndex.get(metric))));
					sequence.problems.add(p);
					++lines;
					if (lines % 10000 == 0) System.out.println(lines + " lines read.");
				} catch (NumberFormatException e)
				{
					log("Error reading problem " + record.get(1) + ": " + e.getMessage());
					e.printStackTrace();
				}
		} catch (IOException e)
		{
			log("Error reading input file: " + e.getMessage());
		}

		for (Sequence sequence : set)
			sequence.problems.sort(Comparator.naturalOrder());

		return true;
	}

	/** Exports the Sequences and calculated Knowledge to the output file.
	 * 
	 * @param output - The output file to export the Sequences to. */
	public static void exportData(File output, ArrayList<ArrayList<String>> data)
	{
		@SuppressWarnings("resource")
		Scanner sc = new Scanner(System.in);
		boolean isWritten = false;
		do
			try
			{
				BufferedWriter bw = new BufferedWriter(new FileWriter(output));
				CSVPrinter printer = new CSVPrinter(bw, CSVFormat.DEFAULT.withDelimiter(';'));
				printer.printRecords(data);
				printer.close();
				isWritten = true;
			} catch (IOException e)
			{
				log("Error creating output file: " + e.getMessage() + "\nPress enter to try again.");
				sc.nextLine();
				// e.printStackTrace();
			}
		while (!isWritten);
	}

	private static void exportData(File output, String[][] data)
	{
		log("Exporting to " + output.getName() + "...");
		ArrayList<ArrayList<String>> list = new ArrayList<ArrayList<String>>();
		for (String[] array : data)
		{
			ArrayList<String> a = new ArrayList<String>();
			for (String string : array)
				a.add(string);
			list.add(a);
		}

		exportData(output, list);
	}

	/** Determines the Knowledge Sequences for each Sequence.
	 * 
	 * @param metric */
	private static void findKnowledgeSequences(Metric metric)
	{
		for (Sequence sequence : allSequences)
			sequence.findKnowledgeSequence(metric);
	}

	/** Determines which Problems should be used to compute Precision. */
	private static void findRepresentativeProblems()
	{
		for (Sequence sequence : allSequences)
		{
			if (settings.getProperty("smooth_rmse").equals("true"))
			{
				Problem best = sequence.finalProblem();
				// Find problem with smallest distance to expected
				for (Problem p : sequence.problems)
					if (Math.abs(p.expectedKnowledge - p.knowledge.mean) < Math.abs(best.expectedKnowledge - best.knowledge.mean)) best = p;

				// Use it and all problems after
				for (int i = sequence.problems.indexOf(best); i < sequence.problems.size(); ++i)
					sequence.problems.get(i).isRepresentative = true;
			} else sequence.finalProblem().isRepresentative = true;
		}
	}

	/** @return The Sequence matching the input ID. Creates a new one if doesn't exist. */
	private static Sequence findSequence(ArrayList<Sequence> set, String id)
	{
		for (Sequence sequence : set)
			if (sequence.name.equals(id)) return sequence;
		Sequence s = new Sequence(id);
		set.add(s);
		return s;
	}

	/** Applies Knowledge Tracing on the expected score to find the Expected Knowledge. */
	private static void knowledgeTracingOnExpected()
	{
		// Stores the real sequences while applying on expected.
		ArrayList<Sequence> sequences = new ArrayList<Sequence>();
		sequences.addAll(allSequences);
		allSequences.clear();

		// Stores the real parameters while applying on expected.
		KTParameters[] params = mainParameters.clone();

		for (Sequence sequence : sequences)
			allSequences.add(sequence.asExpected());

		if (!settings.getProperty("correctness").equals("true")) applyThreshold(threshold);
		applyKnowledgeTracing(null);
		for (int seq = 0; seq < sequences.size(); ++seq)
			for (int prob = 0; prob < sequences.get(seq).problems.size(); ++prob)
			{
				sequences.get(seq).problems.get(prob).expectedKnowledge = allSequences.get(seq).problems.get(prob).knowledge.mean;
				sequences.get(seq).problems.get(prob).isTruthCorrect = allSequences.get(seq).problems.get(prob).isCorrect;
			}

		allSequences.clear();
		allSequences.addAll(sequences);
		expectedParameters = mainParameters;
		mainParameters = params;
	}

	public static void log(String text)
	{
		System.out.println(text);
		log.add(text);
	}

	public static void main(String[] args)
	{
		int iterations = 1;
		double max = -1;
		String out = null;
		if (args.length >= 2) try
		{
			iterations = Integer.parseInt(args[1]);
			max = Double.parseDouble(args[2]);
			out = args[3];
		} catch (Exception e)
		{
			System.out.println("Syntax: [settings-path] [<iteration_count> <max> <result-output>]");
			e.printStackTrace();
			return;
		}

		double threshold = -1;
		double[][] iterationResults = new double[iterations + 1][8];

		for (int i = 0; i < iterations + 1; ++i)
			try
			{
				if (iterations == 1 && i == 1) break;
				System.out.println("ITERATION " + i + " --------------------------------");
				if (iterations != 1) threshold = max * i * 1d / iterations;
				mainMethod(args.length != 0 ? args[0] : "settings.properties", max, threshold);

				int s = 0;
				for (Sequence sequence : allSequences)
					for (Problem problem : sequence.problems)
						if (problem.isCorrect) ++s;
				iterationResults[i][0] = s * 1. / totalProblems;

				s = 0;
				for (Sequence sequence : allSequences)
					for (Problem problem : sequence.problems)
						if (problem.isTruthCorrect) ++s;
				iterationResults[i][1] = s * 1. / totalProblems;

				iterationResults[i][2] = computeRMSE(allSequences, false);
				iterationResults[i][3] = computeInitialRMSE(allSequences);

				iterationResults[i][4] = computePrecision(allSequences, false);
				iterationResults[i][5] = computeInitialPrecision(allSequences);

				iterationResults[i][6] = computeVariation(allSequences, false);
				iterationResults[i][7] = computeInitialVariation(allSequences);
			} catch (Exception e)
			{
				log("Error: " + e.getMessage());
				e.printStackTrace();
			}

		{
			String[][] output = new String[iterations + 2][iterationResults[0].length + 1];
			output[0] = new String[] { "Threshold", "Correct problems", "Correct problems GT", "Knowledge RMSE", "Score RMSE", "Knowledge Precision",
					"Score Precision", "Knowledge Variation", "Score Variation" };
			for (int i = 0; i < iterations + 1; ++i)
			{
				output[i + 1][0] = Utils.toString(max * i * 1d / iterations);
				for (int j = 0; j < iterationResults[0].length; ++j)
					output[i + 1][j + 1] = Utils.toString(iterationResults[i][j]);
			}

			if (out != null) exportData(new File(out), output);
		}

		File f = new File("log.txt");
		try
		{
			BufferedWriter w = new BufferedWriter(new FileWriter(f));
			for (String string : log)
				w.write(string + "\n");
			w.close();
		} catch (IOException e)
		{
			e.printStackTrace();
		}
		log("Done!");
	}

	private static void mainMethod(String propertiesPath, double max, double thresholdin)
	{

		try
		{
			settings = new Properties();
			settings.load(new FileInputStream(new File(propertiesPath)));
			for (String property : new String[] { "aggregation_type", "aggregation_value", "correctness", "cross_validation", "input_file", "metric_threshold",
					"metrics", "output_metrics", "output_params", "output_sequences", "scores", "smooth_rmse", "split" })
				if (!settings.containsKey(property))
				{
					log("Missing setting: " + property);
					return;
				}
		} catch (IOException e)
		{
			log("Error reading settings file: " + e.getMessage());
			return;
		}

		if (!createMetrics(settings.getProperty("metrics"),
				settings.getProperty("aggregation_type").equals("weights") ? settings.getProperty("aggregation_value") : null,
				settings.getProperty("metric_threshold")))
			return;

		File sequences = new File(settings.getProperty("input_file"));
		if (!sequences.exists())
		{
			log("Input file doesn't exist: " + settings.getProperty("input_file"));
			return;
		}

		allSequences = new ArrayList<Sequence>();
		testingSet = new ArrayList<Sequence>();
		learningSet = new ArrayList<Sequence>();
		maxScore = max != -1 ? max
				: settings.getProperty("max_score").equals("null") ? -Double.MAX_VALUE : Double.parseDouble(settings.getProperty("max_score"));
		if (!settings.getProperty("max_score").equals("null") || max != -1) maxSet = true;
		minScore = settings.getProperty("min_score").equals("null") ? Double.MAX_VALUE : Double.parseDouble(settings.getProperty("min_score"));
		if (!settings.getProperty("min_score").equals("null")) minSet = true;
		if (!createSequences(allSequences, sequences)) return;
		// Deprecated: not in use => weights are provided in the settings
		if (settings.getProperty("scores").equals("compute")) computeScores();

		// For each exercise indicate if it is correct based on threshold
		if (!settings.getProperty("correctness").equals("true")) try
		{
			threshold = thresholdin != -1 ? thresholdin : Double.parseDouble(settings.getProperty("correctness"));
		} catch (Exception e)
		{
			log("Incorrect value for score threshold: " + settings.getProperty("correctness"));
			return;
		}

		cleanSequences(settings.getProperty("scores").equals("compute") || settings.getProperty("scores").equals("reduce"),
				Boolean.parseBoolean(settings.getProperty("center_metrics")));

		// For each exercise indicate if it is correct based on threshold
		if (!settings.getProperty("correctness").equals("true") || thresholdin != -1) applyThreshold(threshold);

		// Allow to split each sequence as two sessions
		if (!settings.getProperty("split").equals("false")) try
		{
			log("Splitting sequences...");
			ExplorationSplitter.splitSequences(Integer.parseInt(settings.getProperty("split")));
		} catch (Exception e)
		{
			log("Incorrect value for max noise: " + settings.getProperty("split"));
			return;
		}

		// size of test train sets
		try
		{
			testingSize = (int) Math.ceil(allSequences.size() * Double.parseDouble(settings.getProperty("cross_validation")));
		} catch (Exception e)
		{
			log("Incorrect value for cross validation: " + settings.getProperty("cross_validation"));
			return;
		}
		validations = (int) Math.ceil(allSequences.size() * 1. / testingSize);

		// use the aggregated score (see column 'score' in the sample file)
		applyKnowledgeTracing(null);
		if (settings.getProperty("kt_on_metrics").equals("true")) for (Metric metric : metrics)
			applyKnowledgeTracing(metric);

		// performance index of our model: smoothing the results based on several problems evaluations
		findRepresentativeProblems();

		if (metrics.size() != 0 && settings.getProperty("kt_on_metrics").equals("true")) aggregateMetrics();

		knowledgeTracingOnExpected();

		/** Edit by Nicolas: export when aggregated model is used a CSV file containing all local KT models per metric **/
		if (metrics.size() != 0 && settings.getProperty("kt_on_metrics").equals("true") && !settings.getProperty("output_local_KT").equals("null"))
			exportData(new File(PATH + settings.getProperty("output_local_KT")), outputMetricKTParams(metricsKTParams));

		if (!settings.getProperty("output_params").equals("null")) exportData(new File(PATH + settings.getProperty("output_params")), outputParams());
		if (!settings.getProperty("output_sequences").equals("null")) exportData(new File(PATH + settings.getProperty("output_sequences")), outputProblems());
		if (!settings.getProperty("output_metrics").equals("null")) exportData(new File(PATH + settings.getProperty("output_metrics")), outputMetrics());
		if (!settings.getProperty("output_user_graphs").equals("null"))
			exportData(new File(PATH + settings.getProperty("output_user_graphs")), outputUserGraphs());

	}

	/** Function that outputs a String[][] table, each row describing the KT parameters for one specific metric, in the case of aggregated KT
	 * 
	 * @return */
	private static String[][] outputMetricKTParams(HashMap<Metric, KTParameters[]> metricsKTParams)
	{

		String[][] output = new String[metricsKTParams.size() + 1][11];
		String[] tmp = { "Metric name", "P(L0)", "V(L0)", "P(T)", "V(T)", "P(G)", "V(G)", "P(S)", "V(S)", "P(Ln)", "V(Ln)" };
		output[0] = tmp;

		int cpt = 0;
		for (Metric m : metricsKTParams.keySet())
		{
			cpt++;
			output[cpt][0] = m.name;

			/** Compute KT parameters */
			KTParameters[] param = metricsKTParams.get(m);

			// Mean
			double start = 0, transition = 0, guess = 0, slip = 0;
			for (int i = 0; i < param.length - 2; ++i) // -2 because param.length = 12 but only the 10 first values are useable
			{
				start += param[i].startKnowledge;
				transition += param[i].transition;
				guess += param[i].guess.mean;
				slip += param[i].slip.mean;
			}

			start /= (param.length - 2);
			transition /= (param.length - 2);
			guess /= (param.length - 2);
			slip /= (param.length - 2);

			// Variation
			double sStart = 0, sTransition = 0, sGuess = 0, sSlip = 0;

			for (int i = 0; i < param.length - 2; ++i)
			{
				sStart += Math.pow(param[i].startKnowledge - start, 2);
				sTransition += Math.pow(param[i].transition - transition, 2);
				sGuess += Math.pow(param[i].guess.mean - guess, 2);
				sSlip += Math.pow(param[i].slip.mean - slip, 2);
			}

			sStart = Math.sqrt(sStart / (param.length - 2));
			sTransition = Math.sqrt(sTransition / (param.length - 2));
			sGuess = Math.sqrt(sGuess / (param.length - 2));
			sSlip = Math.sqrt(sSlip / (param.length - 2));

			/** Export these values to string in the output table */
			output[cpt][1] = Utils.toString(start);
			output[cpt][2] = Utils.toString(sStart);

			output[cpt][3] = Utils.toString(transition);
			output[cpt][4] = Utils.toString(sTransition);

			output[cpt][5] = Utils.toString(guess);
			output[cpt][6] = Utils.toString(sGuess);

			output[cpt][7] = Utils.toString(slip);
			output[cpt][8] = Utils.toString(sSlip);

			/** Now let's try to get average(P(Ln)) and variation(P(Ln)) for each metric on the set of sequences */
			double PLn = 0; // Values for P(Ln)
			double PLn2 = 0; // Squared values for P(Ln) used to compute the std dev
			for (Sequence sequence : allSequences)
			{
				PLn += sequence.finalProblem().metricKnowledge.get(m).mean;
				PLn2 += Math.pow(sequence.finalProblem().metricKnowledge.get(m).mean, 2);
			}
			// Averaging
			PLn /= allSequences.size(); // E(X)
			PLn2 /= allSequences.size(); // E(X^2)

			output[cpt][9] = Utils.toString(PLn);
			output[cpt][10] = Utils.toString(Math.sqrt(PLn2 - Math.pow(PLn, 2))); // sigma(x) = sqrt(E(X^2) - E(X)^2)
		} // end for metrics m ...

		return output;
	}

	/** @return The Knowledge associated with each problem. */
	private static String[][] outputMetrics()
	{
		String[][] data = new String[totalProblems * metrics.size() + 1][5];

		data[0][0] = "sequence";
		data[0][1] = "problem";
		data[0][2] = "metric";
		data[0][3] = "input";
		data[0][4] = "learned";

		int current = 1;
		for (Sequence sequence : allSequences)
			for (Problem problem : sequence.problems)
				for (Metric metric : metrics)
				{
					data[current][0] = sequence.name;
					data[current][1] = problem.name;
					data[current][2] = metric.name;
					data[current][3] = Utils.toString(metric.initialDistribution.unreduce(problem.metricScores.get(metric)));
					data[current][4] = Utils.toString(metric.initialDistribution.unreduce(problem.metricKnowledge.get(metric).mean));
					++current;
				}

		return data;
	}

	/** @return The data to output. EDIT by Nicolas 10/20/2017 to output average+- std deviation on last problem KT score P(Ln) */
	private static String[][] outputParams()
	{
		String[][] output = new String[21][2];
		output[0][0] = "param";
		output[0][1] = "value";

		output[1][0] = "P(L0)";
		output[1][1] = Utils.toString(mainParameters[validations].startKnowledge);

		output[2][0] = "P(T)";
		output[2][1] = Utils.toString(mainParameters[validations].transition);

		output[3][0] = "mean(P(G))";
		output[3][1] = Utils.toString(mainParameters[validations].guess.mean);

		output[4][0] = "variation(P(G))";
		output[4][1] = Utils.toString(mainParameters[validations].guess.variation);

		output[5][0] = "mean(P((S))";
		output[5][1] = Utils.toString(mainParameters[validations].slip.mean);

		output[6][0] = "variation(P(S))";
		output[6][1] = Utils.toString(mainParameters[validations].slip.variation);

		output[7][0] = "Knowledge Precision";
		output[7][1] = Utils.toString(computePrecision(allSequences, false));

		output[8][0] = "Knowledge Variation";
		output[8][1] = Utils.toString(computeVariation(allSequences, false));

		output[9][0] = "Aggregated Precision";
		output[9][1] = Utils.toString(computePrecision(allSequences, true));

		output[10][0] = "Aggregated Variation";
		output[10][1] = Utils.toString(computeVariation(allSequences, true));

		output[11][0] = "mean(Ln)";
		double average = computeAverageLearning(allSequences, false);
		output[11][1] = Utils.toString(average);

		output[12][0] = "variation(Ln)";
		output[12][1] = Utils.toString(computeDeviationLearning(allSequences, false, average));

		output[13][0] = "mean(Ln) Aggregated";
		average = metrics.size() == 0 || !settings.getProperty("kt_on_metrics").equals("true") ? 0 : computeAverageLearning(allSequences, true);
		output[13][1] = Utils.toString(average);

		output[14][0] = "variation(Ln) Aggregated";
		output[14][1] = Utils.toString(
				metrics.size() == 0 || !settings.getProperty("kt_on_metrics").equals("true") ? 0 : computeDeviationLearning(allSequences, true, average));

		output[15][0] = "mean(Ln) KT Helpfulness";
		average = computeAverageExpectedLearning(allSequences);
		output[15][1] = Utils.toString(average);

		output[16][0] = "variation(Ln) KT Helpfulness";
		output[16][1] = Utils.toString(computeDeviationExpectedLearning(allSequences, average));

		output[17][0] = "Score Precision";
		output[17][1] = Utils.toString(computeInitialPrecision(allSequences));

		output[18][0] = "Score Variation";
		output[18][1] = Utils.toString(computeInitialVariation(allSequences));

		int s = 0;
		for (Sequence sequence : allSequences)
			if (sequence.problems.get(0).isCorrect) ++s;
		output[19][0] = "Correct starts";
		output[19][1] = Utils.toString(s * 1. / allSequences.size());

		s = 0;
		for (Sequence sequence : allSequences)
			for (Problem problem : sequence.problems)
				if (problem.isCorrect) ++s;
		output[20][0] = "Correct problems";
		output[20][1] = Utils.toString(s * 1. / totalProblems);

		return output;
	}

	/** @return The Knowledge associated with each problem. */
	private static String[][] outputProblems()
	{
		String[][] data = new String[totalProblems + 1][10];

		data[0][0] = "sequence";
		data[0][1] = "problem";
		data[0][2] = "computed_knowledge";
		data[0][3] = "computed_variation";
		data[0][4] = "aggregated_knowledge";
		data[0][5] = "aggregated_variation";
		data[0][6] = "learned";
		data[0][7] = "learned_aggregation";
		data[0][8] = "expected";
		data[0][9] = "sequence_rmse";

		int current = 1;
		double threshold = settings.getProperty("expected_binary").equals("false") ? -1 : Double.parseDouble(settings.getProperty("expected_binary"));
		double rmse;
		ArrayList<Sequence> s = new ArrayList<Sequence>();
		for (Sequence sequence : allSequences)
		{
			s.clear();
			s.add(sequence);
			rmse = computePrecision(s, false);
			for (Problem problem : sequence.problems)
			{
				data[current][0] = sequence.name;
				data[current][1] = problem.name;
				data[current][2] = Utils.toString(problem.knowledge.mean);
				data[current][3] = Utils.toString(problem.knowledge.variation);
				data[current][4] = problem.aggregatedKnowledge == null ? "N/A" : Utils.toString(problem.aggregatedKnowledge.mean);
				data[current][5] = problem.aggregatedKnowledge == null ? "N/A" : Utils.toString(problem.aggregatedKnowledge.variation);
				data[current][6] = threshold == -1 ? Utils.toString(problem.knowledge.mean) : problem.knowledge.mean > threshold ? "1" : "0";
				data[current][7] = threshold == -1 ? problem.aggregatedKnowledge == null ? "N/A" : Utils.toString(problem.aggregatedKnowledge.mean)
						: problem.aggregatedKnowledge.mean > threshold ? "1" : "0";
				data[current][8] = Utils.toString(problem.expectedKnowledge);
				data[current][9] = Utils.toString(rmse);
				++current;
			}
		}

		return data;
	}

	/** @return The file to be processed for SVM. */
	private static String[][] outputSVM(boolean binary)
	{
		int size = 0;
		for (Sequence sequence : allSequences)
			for (Problem problem : sequence.problems)
				if (problem.isRepresentative) ++size;
		String[][] data = new String[size + 1][2 + metrics.size()];

		data[0][0] = "problem";
		for (int i = 0; i < metrics.size(); ++i)
			data[0][i + 1] = metrics.get(i).name;
		data[0][1 + metrics.size()] = "expected";

		int current = 1;
		double threshold = settings.getProperty("expected_binary").equals("false") ? -1 : Double.parseDouble(settings.getProperty("expected_binary"));
		for (Sequence sequence : allSequences)
			for (Problem problem : sequence.problems)
				if (problem.isRepresentative)
				{
					data[current][0] = problem.name;
					for (int i = 0; i < metrics.size(); ++i)
						data[current][i + 1] = Utils.toString(metrics.get(i).initialDistribution.unreduce(problem.metricScores.get(metrics.get(i))));
					data[current][1 + metrics.size()] = binary && threshold != -1 ? problem.expectedKnowledge >= threshold ? "1" : "0"
							: Utils.toString(problem.expectedKnowledge);
					++current;
				}

		return data;
	}

	private static String[][] outputUserGraphs()
	{
		String[][] data = new String[allSequences.size() * 6][];
		String[] score, expected, kt, ktexpected;

		int i = 0;
		for (Sequence sequence : allSequences)
		{
			score = new String[sequence.problems.size() + 1];
			expected = new String[sequence.problems.size() + 1];
			kt = new String[sequence.problems.size() + 1];
			ktexpected = new String[sequence.problems.size() + 1];

			score[0] = "Score";
			expected[0] = "Expected";
			kt[0] = "Knowledge";
			ktexpected[0] = "Expected Knowledge";

			int p = 0;
			for (Problem problem : sequence.problems)
			{
				score[++p] = Utils.toString(problem.score);
				expected[p] = Utils.toString(problem.groundTruth);
				kt[p] = Utils.toString(problem.knowledge.mean);
				ktexpected[p] = Utils.toString(problem.expectedKnowledge);
			}

			data[i++] = new String[] { sequence.name, "Data..." };
			data[i++] = score;
			data[i++] = expected;
			data[i++] = kt;
			data[i++] = ktexpected;
			data[i++] = new String[] {};
		}

		return data;
	}
}
