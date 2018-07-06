package fr.diblois.ckt.util;

import java.util.Scanner;

import fr.diblois.ckt.RedditCKT;

public class PythonLogger implements Runnable
{

	@Override
	public void run()
	{
		System.out.println("started");
		while (RedditCKT.process == null)
			try
			{
				Thread.sleep(10);
			} catch (InterruptedException e)
			{
				e.printStackTrace();
			}

		System.out.println("process is created");
		String line;
		Scanner sc = new Scanner(RedditCKT.process.getInputStream());
		while (!(line = sc.nextLine()).contains("DNNR Finished"))
			System.out.println("[PYTHON]: " + line);
		System.out.println("didn't wait");
		sc.close();
	}

}
