package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {

    public static final String UNKNOWN_WORD = "UUUNKKK";
    public static final String DIGIT_WORD = "DG";
    public static final String START_TOKEN = "<s>";
    public static final String END_TOKEN = "</s>";


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
        int numVectors = 0;
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

        // populate vector matrix
        double[] column = new double[vectorDimension];
        allVecs.setColumn(0, 0, column);
        br = new BufferedReader(new FileReader(vecFilename));
        line = br.readLine();

        int colIdx = 0;

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

	public static void initializeVocab(String vocabFilename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(vocabFilename));

        int rowIdx = 0;
        String line = br.readLine();
        while (line != null) {
            String word = line.trim();

            if (word.length() != 0) {
                wordToNum.put(word, rowIdx);
                numToWord.put(rowIdx, word);
                rowIdx++;
            }

            line = br.readLine();
        }
        br.close();

	}


    public static SimpleMatrix randomInitializeWordVectors(int vectorDimension, int numVectors) {
        allVecs = new SimpleMatrix(vectorDimension, numVectors);

        Random generator = new Random();
        for (int colIdx = 0; colIdx < numVectors; colIdx++) {
            for (int rowIdx = 0; rowIdx < vectorDimension; rowIdx++) {
                double val = generator.nextDouble() * 2 - 1.0; // rand value between -1.0 and 1.0
                allVecs.set(rowIdx, colIdx, val);
            }
        }

        return allVecs;
    }

}
