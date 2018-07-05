package fr.diblois.ckt.data;

/** Knowledge Tracing Fold Iteration Results. */
public class KTFIResults
{

	/** The index of this fold. */
	public final int fold;
	/** The KTTIs done in this KTFI. */
	public final KTTIResults[] kttis;
	/** The RMSE on prediction results. */
	public final double rmse_prediction;

	public KTFIResults(int fold, double rmse_prediction, KTTIResults[] kttis)
	{
		super();
		this.fold = fold;
		this.rmse_prediction = rmse_prediction;
		this.kttis = kttis;
	}

}
