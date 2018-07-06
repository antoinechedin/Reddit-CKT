package fr.diblois.ckt.data;

/** Knowledge Tracing Fold Iteration Results. */
public class KTFIResults
{

	/** Propotion of correct problems in ground truth. */
	public final double correct_problems_gt;
	/** Proportion of correct problems in predicted karma. */
	public final double correct_problems_predicted;
	/** The index of this fold. */
	public final int fold;
	/** Parameters predicted for this iteration. */
	public final KTParameters parameters;
	/** The RMSE on Knowledge Tracing results. */
	public final double rmse_kt;

	public KTFIResults(int fold, KTParameters parameters, double correct_problems_gt, double correct_problems_predicted, double rmse_kt)
	{
		this.fold = fold;
		this.parameters = parameters;
		this.correct_problems_gt = correct_problems_gt;
		this.correct_problems_predicted = correct_problems_predicted;
		this.rmse_kt = rmse_kt;
	}

}