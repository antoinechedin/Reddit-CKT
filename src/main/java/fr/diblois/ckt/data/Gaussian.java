package fr.diblois.ckt.data;

import fr.diblois.ckt.util.Utils;

/** Represents a Gaussian distribution by its mean and variation. */
public class Gaussian
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

	/** @return The input value after applying the center-reduce method. */
	public double reduce(double value)
	{
		return (value - this.mean) / this.variation;
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