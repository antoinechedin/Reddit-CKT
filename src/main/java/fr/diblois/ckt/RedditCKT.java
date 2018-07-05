package fr.diblois.ckt;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Properties;

public class RedditCKT
{

	private static final ArrayList<String> log = new ArrayList<>();
	public static final Properties settings = new Properties();

	private static void finish()
	{
		try
		{
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File("log.txt")));
			for (String message : log)
				writer.write(message + "\n");
			writer.close();
		} catch (IOException e)
		{
			log("File log.txt couldn't be opened: " + e.getMessage());
			e.printStackTrace();
		}
	}

	/** Prints a message and adds it to the log. */
	public static void log(String message)
	{
		log.add(message);
		System.out.println(message);
	}

	public static void main(String[] args)
	{
		if (args.length >= 1 && args[0].equals("help"))
		{
			System.out.println("Usage: [settings_file_path]");
			return;
		}

		log("Starting Reddit CKT.");

		String settingsPath = args.length == 0 ? "settings.properties" : args[0];

		try
		{
			settings.load(new FileInputStream(new File(settingsPath)));
		} catch (FileNotFoundException e)
		{
			log("File " + settingsPath + " not found: " + e.getMessage());
		} catch (IOException e)
		{
			log("File " + settingsPath + " couldn't be opened: " + e.getMessage());
		}

		
		finish();
	}

}
