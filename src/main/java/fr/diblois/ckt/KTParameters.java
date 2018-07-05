package fr.diblois.ckt;

/** The set of Parameters used in Knowledge Tracing. */
public class KTParameters
{

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
