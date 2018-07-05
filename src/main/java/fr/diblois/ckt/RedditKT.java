package fr.diblois.ckt;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

public class RedditKT
{

	public static final Properties settings = new Properties();

	public static void main(String[] args)
	{
		if (args.length >= 1 && args[0].equals("help"))
		{
			System.out.println("Usage: [settings_file_path]");
			return;
		}

		String settingsPath = args.length == 0 ? "settings.properties" : args[0];

		try
		{
			settings.load(new FileInputStream(new File(settingsPath)));
		} catch (FileNotFoundException e)
		{
			System.err.println("File " + settingsPath + " not found: " + e.getMessage());
		} catch (IOException e)
		{
			System.err.println("File " + settingsPath + " couldn't be opened: " + e.getMessage());
		}
	}

}
