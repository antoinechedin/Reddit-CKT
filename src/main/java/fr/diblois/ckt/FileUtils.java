package fr.diblois.ckt;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import fr.diblois.ckt.util.Problem;
import fr.diblois.ckt.util.Sequence;

public class FileUtils
{

	/** Exports the Sequences and calculated Knowledge to the output file.
	 * 
	 * @param output - The output file to export the Sequences to. */
	public static void exportData(File output, ArrayList<ArrayList<String>> data)
	{
		@SuppressWarnings("resource")
		Scanner sc = new Scanner(System.in);
		boolean isWritten = false;
		do
			try
			{
				BufferedWriter bw = new BufferedWriter(new FileWriter(output));
				CSVPrinter printer = new CSVPrinter(bw, CSVFormat.DEFAULT.withDelimiter(';'));
				printer.printRecords(data);
				printer.close();
				isWritten = true;
			} catch (IOException e)
			{
				RedditCKT.log("Error creating output file: " + e.getMessage() + "\nPress enter to try again.");
				sc.nextLine();
				// e.printStackTrace();
			}
		while (!isWritten);
	}

	/** Exports the input data into the input file.
	 * 
	 * @param data - Matrix representing the lines and cells of a CSV file's data. */
	public static void exportData(File output, String[][] data)
	{
		RedditCKT.log("Exporting to " + output.getName() + "...");
		ArrayList<ArrayList<String>> list = new ArrayList<ArrayList<String>>();
		for (String[] array : data)
		{
			ArrayList<String> a = new ArrayList<String>();
			for (String string : array)
				a.add(string);
			list.add(a);
		}

		exportData(output, list);
	}

	/** Reads the input data from the dataset directory. */
	static void readInputData()
	{
		RedditCKT.log("Reading data.");
		HashMap<String, HashSet<Problem>> problems = new HashMap<>();
		ArrayList<ArrayList<String>> subsets = new ArrayList<>();

		for (int fold = 0; fold < RedditCKT.folds; ++fold)
		{
			subsets.add(new ArrayList<>());
			try
			{
				CSVParser parser = new CSVParser(
						new BufferedReader(new FileReader(new File(RedditCKT.dataset_directory + File.separator + "part_" + fold + ".csv"))),
						CSVFormat.DEFAULT.withFirstRecordAsHeader());

				for (CSVRecord record : parser)
				{
					try
					{
						Problem p = Problem.create(record);
						if (!problems.containsKey(p.sequence)) problems.put(p.sequence, new HashSet<>());
						if (!subsets.get(subsets.size() - 1).contains(p.sequence)) subsets.get(subsets.size() - 1).add(p.sequence);
						problems.get(p.sequence).add(p);
					} catch (Exception e)
					{
						e.printStackTrace();
						RedditCKT.log("Error while initializing problem: " + e.getMessage());
						RedditCKT.log("Problem: " + record.toString());
					}
				}

				parser.close();
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}

		for (String sequence : problems.keySet())
		{
			Sequence s = new Sequence(sequence);
			s.problems.addAll(problems.get(sequence));
			s.problems.sort(Comparator.naturalOrder());
			RedditCKT.dataset.add(s);
		}

		for (ArrayList<String> subset : subsets)
		{
			ArrayList<Sequence> s = new ArrayList<>();
			for (String seq : subset)
				s.add(RedditCKT.getSequence(seq));
			RedditCKT.subsets.add(s);
		}

		RedditCKT.dataset.sort(Comparator.naturalOrder());
	}

	/** Reads predictions made by the python script. */
	public static void readPredictions(String predictionsPath)
	{
		RedditCKT.log("Reading predictions.");
		try
		{
			CSVParser parser = new CSVParser(new BufferedReader(new FileReader(new File(predictionsPath))), CSVFormat.DEFAULT.withFirstRecordAsHeader());
			for (CSVRecord record : parser)
			{
				String id = record.get(RedditCKT.column_problem);
				Problem p = RedditCKT.getProblem(id);
				String pred = record.get(RedditCKT.column_karma_predicted);
				p.prediction = Double.valueOf(pred);
			}
			parser.close();
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	/** Reads the parameters contained in the properties file. */
	static void readSettings(String settingsPath)
	{
		RedditCKT.log("Reading settings.");
		try
		{
			RedditCKT.settings.load(new FileInputStream(new File(settingsPath)));
		} catch (FileNotFoundException e)
		{
			RedditCKT.log("File " + settingsPath + " not found: " + e.getMessage());
		} catch (IOException e)
		{
			RedditCKT.log("File " + settingsPath + " couldn't be opened: " + e.getMessage());
		}

		RedditCKT.dataset_directory = RedditCKT.settings.getProperty("dataset_directory");
		RedditCKT.results_directory = RedditCKT.settings.getProperty("results_directory");
		RedditCKT.column_order = RedditCKT.settings.getProperty("column_order");
		RedditCKT.column_problem = RedditCKT.settings.getProperty("column_problem");
		RedditCKT.column_score = RedditCKT.settings.getProperty("column_score");
		RedditCKT.column_sequence = RedditCKT.settings.getProperty("column_sequence");
		RedditCKT.column_karma_predicted = RedditCKT.settings.getProperty("column_karma_predicted");
		RedditCKT.predictions_file = RedditCKT.settings.getProperty("predictions_file");
		RedditCKT.script = RedditCKT.settings.getProperty("execute_script");
		if (RedditCKT.script != null && RedditCKT.script.equals("null")) RedditCKT.script = null;

		try
		{
			RedditCKT.folds = Integer.parseInt(RedditCKT.settings.getProperty("fold_count"));
		} catch (NumberFormatException e)
		{
			RedditCKT.log("Invalid fold count: " + RedditCKT.settings.getProperty("fold_count"));
			return;
		}
		try
		{
			RedditCKT.threshold_count = Integer.parseInt(RedditCKT.settings.getProperty("threshold_count"));
		} catch (NumberFormatException e)
		{
			RedditCKT.log("Invalid fold count: " + RedditCKT.settings.getProperty("threshold_count"));
			return;
		}

		try
		{
			String min = RedditCKT.settings.getProperty("min_value");
			RedditCKT.minFixed = !min.equals("null");
			if (RedditCKT.minFixed) RedditCKT.minValue = Double.parseDouble(min);
		} catch (NumberFormatException e)
		{
			RedditCKT.log("Invalid min value: " + RedditCKT.settings.getProperty("min_value"));
			return;
		}
		try
		{
			String max = RedditCKT.settings.getProperty("max_value");
			RedditCKT.maxFixed = !max.equals("null");
			if (RedditCKT.maxFixed) RedditCKT.maxValue = Double.parseDouble(max);
		} catch (NumberFormatException e)
		{
			RedditCKT.log("Invalid max value: " + RedditCKT.settings.getProperty("max_value"));
			return;
		}
	}

}
