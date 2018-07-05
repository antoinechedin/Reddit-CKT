package fr.diblois.ckt.util;

import java.util.HashMap;

import org.apache.commons.csv.CSVRecord;

import fr.diblois.ckt.data.Gaussian;

public class Problem implements Comparable<Problem>
{

	public static Problem create(CSVRecord record) throws Exception
	{
		// TODO read metrics
		Problem p = new Problem(record.get("problem"), record.get("sequence"), Integer.parseInt(record.get("order")));
		p.karma = Double.parseDouble(record.get("karma"));
		return p;
	}

	/** The Knowledge aggregated from all metrics (only used for representative Problems). */
	public Gaussian aggregatedKnowledge;
	/** The knowledge expected after this Problem. */
	public double expectedKnowledge;
	/** Index of the Problem in the Exploration. */
	public final int index;
	/** True if Problem is considered correct for the ideal sequence. */
	public boolean isCorrect;
	/** True if this Problem should be used when computing precision. */
	public boolean isRepresentative = false;
	/** True if Problem's ground truth is considered correct for the ideal sequence. */
	public boolean isTruthCorrect;
	/** The karma expected for this Problem. */
	public double karma;
	/** The Knowledge computed after this Problem. */
	public Gaussian knowledge;
	/** True if Problem is considered correct for the ideal sequence for each metric. */
	public final HashMap<Metric, Boolean> metricCorrectness;
	/** The knowledge for each individual metric. */
	public final HashMap<Metric, Gaussian> metricKnowledge;
	/** The score for each individual metric. */
	public final HashMap<Metric, Double> metricScores;
	/** Problem name. */
	public final String name;
	/** The prediction of this problem's karma. */
	public double prediction;
	/** Sequence this Problem is part of. */
	public final String sequence;

	public Problem(String name, String sequence)
	{
		this(name, sequence, -1);
	}

	public Problem(String name, String sequence, int index)
	{
		this.name = name;
		this.sequence = sequence;
		this.index = index;
		this.metricScores = new HashMap<Metric, Double>();
		this.metricKnowledge = new HashMap<Metric, Gaussian>();
		this.metricCorrectness = new HashMap<Metric, Boolean>();
	}

	/** @return A copy of this Problem, with scores as expected. */
	public Problem asExpected()
	{
		Problem copy = new Problem(this.name, this.sequence, this.index);
		copy.prediction = this.karma;
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
