package fr.diblois.ckt.data;

/** Knowledge Tracing Fold Iteration Results.<br>
 * Represents the results of a single fold in the cross validation, for a fixed threshold. */
public class KTFIResults
{

	/** Propotion of correct problems in ground truth. */
	public final double correct_problems_gt;
	/** Proportion of correct problems in predicted karma. */
	public final double correct_problems_predicted;
	/** The index of this fold. */
	public final int fold;
	/** The MAE on Knowledge Tracing results. */
	public final double mae_kt;
	/** Parameters predicted for this iteration. */
	public final KTParameters parameters;
	/** Parameters predicted on ground truth for this iteration. */
	public final KTParameters parametersGT;
	/** The RMSE on Knowledge Tracing results. */
	public final double rmse_kt;

	public KTFIResults(int fold, KTParameters parametersGT, KTParameters parameters, double correct_problems_gt, double correct_problems_predicted,
			double rmse_kt, double mae_kt)
	{
		this.fold = fold;
		this.parametersGT = parametersGT;
		this.parameters = parameters;
		this.correct_problems_gt = correct_problems_gt;
		this.correct_problems_predicted = correct_problems_predicted;
		this.rmse_kt = rmse_kt;
		this.mae_kt = mae_kt;
	}

}
