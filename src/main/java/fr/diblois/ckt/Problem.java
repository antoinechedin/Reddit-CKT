package fr.diblois.ckt;

import java.util.HashMap;

public class Problem implements Comparable<Problem>
{

	/** The Knowledge aggregated from all metrics (only used for final Problems). */
	KTParameters.Gaussian aggregatedKnowledge;
	/** The knowledge expected after this Problem. */
	double expectedKnowledge;
	/** The ground truth expected for this Problem. */
	double groundTruth;
	/** Index of the Problem in the Exploration. */
	int index;
	/** True if Problem is considered correct for the ideal sequence. */
	boolean isCorrect;
	/** True if Problem is considered correct for the ideal sequence (temporary variable for metrics). */
	boolean isCorrectTemp;
	/** True if this Problem should be used when computing precision. */
	boolean isRepresentative = false;
	/** True if Problem's ground truth is considered correct for the ideal sequence. */
	boolean isTruthCorrect;
	/** The Knowledge computed after this Problem. */
	KTParameters.Gaussian knowledge;
	/** The knowledge for each individual metric. */
	public final HashMap<Metric, KTParameters.Gaussian> metricKnowledge;
	/** The score for each individual metric. */
	public final HashMap<Metric, Double> metricScores;
	/** Problem name. */
	public final String name;
	/** The focusness score of this problem. */
	double score;

	public Problem(String name)
	{
		this(name, -1);
	}

	public Problem(String name, int index)
	{
		this.name = name;
		this.index = index;
		this.metricScores = new HashMap<Metric, Double>();
		this.metricKnowledge = new HashMap<Metric, KTParameters.Gaussian>();
	}

	/** @return A copy of this Problem, with scores as expected. */
	public Problem asExpected()
	{
		Problem copy = new Problem(this.name, this.index);
		copy.score = this.groundTruth;
		copy.index = this.index;
		copy.isCorrect = this.isCorrect;
		return copy;
	}

	@Override
	public int compareTo(Problem o)
	{
		if (this.index == -1) return this.name.compareTo(o.name);
		return Integer.compare(this.index, o.index);
	}
}
