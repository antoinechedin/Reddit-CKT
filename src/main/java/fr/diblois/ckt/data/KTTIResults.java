package fr.diblois.ckt.data;

/** Knowledge Tracing Threshold Iteration Results. */
public class KTTIResults
{

	private double kt_mae = -1;
	private double kt_rmse = -1;
	/** The KTFIs done in this KTTI. */
	public final KTFIResults[] ktfis;
	/** The threshold that was used during this iteration. */
	public final double threshold;

	public KTTIResults(double threshold, KTFIResults[] ktfis)
	{
		this.threshold = threshold;
		this.ktfis = ktfis;
	}

	public double corrects()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.correct_problems_predicted;
		return avg * 1. / this.ktfis.length / 100;
	}

	public double correctsGT()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.correct_problems_gt;
		return avg * 1. / this.ktfis.length / 100;
	}

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

	public double pg_avg()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parameters.guess.mean;
		return avg / this.ktfis.length;
	}

	public double pg_stdev()
	{
		double avg = this.pg_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parameters.guess.mean - avg, 2);
		return stdev / this.ktfis.length;
	}

	public double pl0_avg()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parameters.startKnowledge;
		return avg / this.ktfis.length;
	}

	public double pl0_stdev()
	{
		double avg = this.pl0_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parameters.startKnowledge - avg, 2);
		return stdev / this.ktfis.length;
	}

	public double ps_avg()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parameters.slip.mean;
		return avg / this.ktfis.length;
	}

	public double ps_stdev()
	{
		double avg = this.ps_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parameters.slip.mean - avg, 2);
		return stdev / this.ktfis.length;
	}

	public double pt_avg()
	{
		double avg = 0;
		for (KTFIResults ktfi : ktfis)
			avg += ktfi.parameters.transition;
		return avg / this.ktfis.length;
	}

	public double pt_stdev()
	{
		double avg = this.pt_avg();
		double stdev = 0;
		for (KTFIResults ktfi : ktfis)
			stdev += Math.pow(ktfi.parameters.transition - avg, 2);
		return stdev / this.ktfis.length;
	}

}