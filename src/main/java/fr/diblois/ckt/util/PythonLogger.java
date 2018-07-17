package fr.diblois.ckt.util;

import java.util.Scanner;

import fr.diblois.ckt.RedditCKT;

public class PythonLogger implements Runnable
{

	@Override
	public void run()
	{
		while (RedditCKT.process == null)
			try
			{
				Thread.sleep(10);
			} catch (InterruptedException e)
			{
				e.printStackTrace();
			}

		String line;
		Scanner sc = new Scanner(RedditCKT.process.getInputStream());

		while (!RedditCKT.processFinished)
		{
			if (sc.hasNextLine())
			{
				line = sc.nextLine();
				if (line.contains("DNNR Finished")) RedditCKT.processFinished = true;
				else RedditCKT.log("[PYTHON]: " + line);
			} else try
			{
				Thread.sleep(100);
			} catch (InterruptedException e)
			{
				e.printStackTrace();
			}
		}
		sc.close();
	}

}
