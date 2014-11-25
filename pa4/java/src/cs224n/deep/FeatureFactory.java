package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
	    // TODO: you'd want to handle sentence boundaries
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}
 
 
	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		if (allVecs!=null) return allVecs;

        // get vector matrix size
        int numVectors = 1; // start from 1 because of UNKNOWN_WORD column
        BufferedReader br = new BufferedReader(new FileReader(vecFilename));
        String line = br.readLine();
        int vectorDimension = line.trim().split(" ").length;

        while (line != null) {
            if (line.trim().length() != 0) {
                numVectors++;
            }
            line = br.readLine();
        }
        br.close();

        allVecs = new SimpleMatrix(vectorDimension, numVectors);
//        System.out.println("N = " + allVecs.numRows() + ", V = " + allVecs.numCols());

        // populate vector matrix
        double[] column = new double[vectorDimension];
        allVecs.setColumn(0, 0, column);
        br = new BufferedReader(new FileReader(vecFilename));
        line = br.readLine();

        int colIdx = 1;

        while (line != null) {
            if (line.trim().length() == 0) {
                continue;
            }
            String[] tokens = line.trim().split(" ");

            if (tokens.length != vectorDimension) {
                System.out.println("Tokens length doesn't match");
            }

            for (int rowIdx = 0; rowIdx < vectorDimension; rowIdx++) {
                double val = Double.valueOf(tokens[rowIdx]);
                allVecs.set(rowIdx, colIdx, val);
            }

            colIdx++;
            line = br.readLine();
        }
        br.close();

		return null;
	}
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

    public static String UNKNOWN_WORD = "UNKNOWN_WORD";

	public static void initializeVocab(String vocabFilename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(vocabFilename));

        // Add the unknown word to the lookups
        wordToNum.put(UNKNOWN_WORD, 0);
        numToWord.put(0, UNKNOWN_WORD);

        int rowIdx = 1;
        String line = br.readLine();
        while (line != null) {
            if (line.trim().length() == 0) {
                continue;
            }
            wordToNum.put(line, rowIdx);
            numToWord.put(rowIdx, line);
            rowIdx++;
            line = br.readLine();
        }
        br.close();
	}
 








}
