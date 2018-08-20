package fr.diblois.ckt.util;

import org.apache.commons.csv.CSVRecord;

import fr.diblois.ckt.RedditCKT;
import fr.diblois.ckt.data.Gaussian;

public class Problem implements Comparable<Problem>
{

	public static Problem create(CSVRecord record) throws Exception
	{
		Problem p = new Problem(record.get(RedditCKT.column_problem), record.get(RedditCKT.column_sequence),
				Double.valueOf(record.get(RedditCKT.column_order)).intValue());
		p.karma = Double.valueOf(record.get(RedditCKT.column_score));
		return p;
	}

	/** The knowledge expected after this Problem. */
	public Gaussian expectedKnowledge;
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
	}

	@Override
	public int compareTo(Problem o)
	{
		if (this.index == -1) return this.name.compareTo(o.name);
		return Integer.compare(this.index, o.index);
	}
}
