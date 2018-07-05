package fr.diblois.ckt.data;

import fr.diblois.ckt.util.Utils;

/** The set of Parameters used in Knowledge Tracing. */
public class KTParametersStats
{

	/** P(G) */
	public final double guess;
	/** P(S) */
	public final double slip;
	/** P(L0) */
	public final double startKnowledge;
	/** P(T) */
	public final double transition;

	public KTParametersStats(double startKnowledge, double transition, double guess, double slip)
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
		sb.append(Utils.toString(this.guess));
		sb.append(", S: ");
		sb.append(Utils.toString(this.slip));
		return sb.toString();
	}

}
