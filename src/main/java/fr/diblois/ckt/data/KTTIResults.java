package fr.diblois.ckt.data;

/** Knowledge Tracing Threshold Iteration Results.<br>
 * Represents the results of all folds in the cross validation, for a fixed threshold. */
public class KTTIResults
{

	/** The MAE on KT. -1 if not computed yet. */
	private double kt_mae = -1;
	/** The RMSE on KT. -1 if not computed yet. */
	private double kt_rmse = -1;
	/** The KTFIs done in this KTTI. */
	public final KTFIResults[] ktfis;
	/** The average parameters for this Threshold. */
	public final KTParametersStats parametersAvg;
	/** The average parameters for ground truth for this Threshold. */
	public final KTParametersStats parametersGTAvg;
	/** The standard deviation for parameters for ground truth for this Threshold. */
	public final KTParametersStats parametersGTStdev;
	/** The standard deviation for parameters for this Threshold. */
	public final KTParametersStats parametersStdev;
	/** The threshold that was used during this iteration. */
	public final double threshold;

	public KTTIResults(double threshold, KTFIResults[] ktfis)
	{
		this.threshold = threshold;
		this.ktfis = ktfis;

		this.parametersAvg = new KTParametersStats(this.pl0_avg(), this.pt_avg(), this.pg_avg(), this.ps_avg());
		this.parametersStdev = new KTParametersStats(this.pl0_stdev(), this.pt_stdev(), this.pg_stdev(), this.ps_stdev());
		this.parametersGTAvg = new KTParametersStats(this.pl0_gt(), this.pt_gt(), this.pg_gt(), this.ps_gt());
		this.parametersGTStdev = new KTParametersStats(this.pl0_gtstdev(), this.pt_gtstdev(), this.pg_gtstdev(), this.ps_gtstdev());
	}

	/** @return The proportion of correct problems in for this Threshold. */
	public double corrects()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.correct_problems_predicted;
		return avg * 1. / this.ktfis.length / 100;
	}

	/** @return The proportion of correct problems in ground truth in for this Threshold. */
	public double correctsGT()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.correct_problems_gt;
		return avg * 1. / this.ktfis.length / 100;
	}

	/** @return The MAE on KT for this Threshold. */
	public double ktMAE()
	{
		if (this.kt_mae == -1)
		{
			double avg = 0;
			for (KTFIResults ktfi : ktfis)
				avg += ktfi.mae_kt;
			this.kt_mae = avg / this.ktfis.length;
		}
		return this.kt_mae;
	}

	/** @return The RMSE on KT for this Threshold. */
	public double ktRMSE()
	{
		if (this.kt_rmse == -1)
		{
			double avg = 0;
			for (KTFIResults ktfi : ktfis)
				avg += ktfi.rmse_kt;
			this.kt_rmse = avg / this.ktfis.length;
		}
		return this.kt_rmse;
	}

	/** @return The average value for P(G) for this Threshold. */
	private double pg_avg()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parameters.guess.mean;
		return avg / this.ktfis.length;
	}

	/** @return The average value for P(G) on ground truth for this Threshold. */
	private double pg_gt()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parametersGT.guess.mean;
		return avg / this.ktfis.length;
	}

	/** @return The standard deviation for P(G) on ground truth for this Threshold. */
	private double pg_gtstdev()
	{
		double avg = this.pg_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parametersGT.guess.mean - avg, 2);
		return stdev / this.ktfis.length;
	}

	/** @return The standard deviation for P(G) for this Threshold. */
	private double pg_stdev()
	{
		double avg = this.pg_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parameters.guess.mean - avg, 2);
		return stdev / this.ktfis.length;
	}

	/** @return The average value for P(L0) for this Threshold. */
	private double pl0_avg()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parameters.startKnowledge;
		return avg / this.ktfis.length;
	}

	/** @return The average value for P(L0) on ground truth for this Threshold. */
	private double pl0_gt()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parametersGT.startKnowledge;
		return avg / this.ktfis.length;
	}

	/** @return The standard deviation for P(L0) on ground truth for this Threshold. */
	private double pl0_gtstdev()
	{
		double avg = this.pl0_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parametersGT.startKnowledge - avg, 2);
		return stdev / this.ktfis.length;
	}

	/** @return The standard deviation for P(L0) for this Threshold. */
	private double pl0_stdev()
	{
		double avg = this.pl0_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parameters.startKnowledge - avg, 2);
		return stdev / this.ktfis.length;
	}

	/** @return The average value for P(S) for this Threshold. */
	private double ps_avg()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parameters.slip.mean;
		return avg / this.ktfis.length;
	}

	/** @return The average value for P(S) on ground truth for this Threshold. */
	private double ps_gt()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parametersGT.slip.mean;
		return avg / this.ktfis.length;
	}

	/** @return The standard deviation for P(S) on ground truth for this Threshold. */
	private double ps_gtstdev()
	{
		double avg = this.ps_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parametersGT.slip.mean - avg, 2);
		return stdev / this.ktfis.length;
	}

	/** @return The standard deviation for P(S) for this Threshold. */
	private double ps_stdev()
	{
		double avg = this.ps_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parameters.slip.mean - avg, 2);
		return stdev / this.ktfis.length;
	}

	/** @return The average value for P(T) for this Threshold. */
	private double pt_avg()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parameters.transition;
		return avg / this.ktfis.length;
	}

	/** @return The average value for P(T) on ground truth for this Threshold. */
	private double pt_gt()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parametersGT.transition;
		return avg / this.ktfis.length;
	}

	/** @return The standard deviation for P(T) on ground truth for this Threshold. */
	private double pt_gtstdev()
	{
		double avg = this.pt_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parametersGT.transition - avg, 2);
		return stdev / this.ktfis.length;
	}

	/** @return The standard deviation for P(T) for this Threshold. */
	private double pt_stdev()
	{
		double avg = this.pt_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parameters.transition - avg, 2);
		return stdev / this.ktfis.length;
	}

}