package fr.diblois.ckt;

/** The set of Parameters used in Knowledge Tracing. */
public class KTParameters
{
	/** Represents a Gaussian distribution by its mean and variation. */
	public static class Gaussian
	{

		/** The Gaussian mean. */
		public final double mean;
		/** The Gaussian variation. */
		public final double variation;

		public Gaussian()
		{
			this(0, 1);
		}

		public Gaussian(double mean)
		{
			this(mean, 1);
		}

		public Gaussian(double mean, double variation)
		{
			this.mean = mean;
			this.variation = variation;
		}

		/** @return A random value from this Gaussian distribution. */
		public double next()
		{
			double next = Utils.random.nextGaussian() * this.variation + this.mean;
			return next < 0 ? 0 : next > 1 ? 1 : next;
		}

		@Override
		public String toString()
		{
			return "(" + Utils.toString(this.mean) + ", " + Utils.toString(this.variation) + ")";
		}

		/** Apologies for the non-scientific method name.
		 * 
		 * @return The input value after reverting the center-reduce method. */
		public double unreduce(double value)
		{
			return value * this.variation + this.mean;
		}

	}

	/** P(G) */
	public final Gaussian guess;
	/** P(S) */
	public final Gaussian slip;
	/** P(L0) */
	public final double startKnowledge;
	/** P(T) */
	public final double transition;

	public KTParameters(double startKnowledge, double transition, Gaussian guess, Gaussian slip)
	{
		super();
		this.startKnowledge = startKnowledge;
		this.transition = transition;
		this.guess = guess;
		this.slip = slip;
	}

	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("L0: ");
		sb.append(Utils.toString(this.startKnowledge));
		sb.append(", T: ");
		sb.append(Utils.toString(this.transition));
		sb.append(", G: ");
		sb.append(this.slip.toString());
		sb.append(", S: ");
		sb.append(this.guess.toString());
		return sb.toString();
	}

}