package fr.diblois.ckt;

import java.util.ArrayList;

import fr.diblois.ckt.util.Problem;
import fr.diblois.ckt.util.Sequence;

public class Stats
{

	public static double computeKarmaRMSE()
	{
		double rmse = 0;
		int problems = 0;// Number of problems in current sequence
		/* for (Sequence sequence : RedditCKT.dataset) { problems = sequence.problems.size(); for (Problem problem : sequence.problems) rmse += Math.pow(problem.karma - problem.prediction, 2) / problems; } */

		for (Sequence sequence : RedditCKT.dataset)
			for (Problem problem : sequence.problems)
			{
				rmse += Math.pow(problem.karma - problem.prediction, 2);
				++problems;
			}

		return Math.sqrt(rmse / problems); // RedditCKT.dataset.size());
	}

	/** @return The Precision for the Knowledge for the input sequences. */
	public static double computeKTRMSE(ArrayList<Sequence> sequences)
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

					precision += Math.pow(problem.expectedKnowledge.mean - problem.knowledge.mean, 2) / problems;
				}
		}

		return Math.sqrt(precision / sequences.size());
	}

}
