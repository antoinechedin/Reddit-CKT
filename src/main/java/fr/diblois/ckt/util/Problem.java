package fr.diblois.ckt.util;

import java.util.HashMap;

import fr.diblois.ckt.data.Gaussian;

public class Problem implements Comparable<Problem>
{

	/** The Knowledge aggregated from all metrics (only used for representative Problems). */
	Gaussian aggregatedKnowledge;
	/** The knowledge expected after this Problem. */
	double expectedKnowledge;
	/** The ground truth expected for this Problem. */
	double groundTruth;
	/** Index of the Problem in the Exploration. */
	int index;
	/** True if Problem is considered correct for the ideal sequence. */
	boolean isCorrect;
	/** True if this Problem should be used when computing precision. */
	boolean isRepresentative = false;
	/** True if Problem's ground truth is considered correct for the ideal sequence. */
	boolean isTruthCorrect;
	/** The karma of this problem. */
	double karma;
	/** The Knowledge computed after this Problem. */
	Gaussian knowledge;
	/** True if Problem is considered correct for the ideal sequence for each metric. */
	public final HashMap<Metric, Boolean> metricCorrectness;
	/** The knowledge for each individual metric. */
	public final HashMap<Metric, Gaussian> metricKnowledge;
	/** The score for each individual metric. */
	public final HashMap<Metric, Double> metricScores;
	/** Problem name. */
	public final String name;

	public Problem(String name)
	{
		this(name, -1);
	}

	public Problem(String name, int index)
	{
		this.name = name;
		this.index = index;
		this.metricScores = new HashMap<Metric, Double>();
		this.metricKnowledge = new HashMap<Metric, Gaussian>();
		this.metricCorrectness = new HashMap<Metric, Boolean>();
	}

	/** @return A copy of this Problem, with scores as expected. */
	public Problem asExpected()
	{
		Problem copy = new Problem(this.name, this.index);
		copy.karma = this.groundTruth;
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
