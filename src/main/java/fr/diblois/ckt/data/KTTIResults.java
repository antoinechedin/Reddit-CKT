package fr.diblois.ckt.data;

/** Knowledge Tracing Threshold Iteration Results. */
public class KTTIResults
{

	/** Propotion of correct problems in ground truth. */
	public final double correct_problems_gt;
	/** Proportion of correct problems in predicted karma. */
	public final double correct_problems_predicted;
	/** Parameters predicted for this iteration. */
	public final KTParameters parameters;
	/** The RMSE on Knowledge Tracing results. */
	public final double rmse_kt;
	/** The threshold that was used during this iteration. */
	public final double threshold;

	public KTTIResults(KTParameters parameters, double threshold, double correct_problems_gt, double correct_problems_predicted, double rmse_kt)
	{
		this.parameters = parameters;
		this.threshold = threshold;
		this.correct_problems_gt = correct_problems_gt;
		this.correct_problems_predicted = correct_problems_predicted;
		this.rmse_kt = rmse_kt;
	}

}
