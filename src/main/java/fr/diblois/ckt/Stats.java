package fr.diblois.ckt;

import fr.diblois.ckt.util.Problem;
import fr.diblois.ckt.util.Sequence;

public class Stats
{

	public static double computeKarmaRMSE()
	{
		double rmse = 0;
		int problems;// Number of representative problems in current sequence
		for (Sequence sequence : RedditCKT.dataset)
		{
			problems = -1;
			for (Problem problem : sequence.problems)
				if (problem.isRepresentative)
				{
					if (problems == -1) problems = sequence.problems.size() - sequence.problems.indexOf(problem);

					rmse += Math.pow(sequence.finalProblem().expectedKnowledge - problem.knowledge.mean, 2) / problems;
				}
		}

		return Math.sqrt(rmse / RedditCKT.dataset.size());
	}

}
