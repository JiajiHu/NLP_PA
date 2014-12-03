package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, U, biasVec, Wout;

    private HashMap<String, Integer> wordToNum;
    private HashMap<String, String> predictions;

	public int windowSize, wordSize, hiddenSize;

    public static String[] predictionLabels = new String[]{"O", "LOC", "MISC", "ORG", "PER"};
    public static int NUM_PREDICTION_CLASSES = 5;


    public static final String BEGIN_TOKEN = "<s>";
    public static final String END_TOKEN = "</s>";

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
        L = FeatureFactory.allVecs;
        wordToNum = FeatureFactory.wordToNum;
        predictions = new HashMap<String, String>();

		windowSize = _windowSize;
        hiddenSize = _hiddenSize;
        wordSize = L.numRows();

	}

    public HashMap<String, String> getPredictions() {
        return predictions;
    }

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
        double epsilon = Math.sqrt(6) / Math.sqrt((double) (windowSize * wordSize + hiddenSize));

        // initialize bias vector randomly to avoid overfitting
        W = SimpleMatrix.random(hiddenSize, wordSize * (windowSize  + 1), -epsilon, epsilon, new Random());
        U = SimpleMatrix.random(NUM_PREDICTION_CLASSES, hiddenSize + 1, -epsilon, epsilon, new Random());

	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> trainData ){
        for (int datumIndex = 0; datumIndex < trainData.size(); datumIndex++) {
            // forward propagation
            SimpleMatrix vectorP = forwardProp(trainData, datumIndex);

            // TODO: Backward propagation


        }
	}

	
	public void test(List<Datum> testData){
        for (int datumIndex = 0; datumIndex < testData.size(); datumIndex++) {
            // forward propagation
            SimpleMatrix vectorP = forwardProp(testData, datumIndex);
            double[] vector = vectorP.extractVector(false, 0).getMatrix().getData();

            if (vector.length != NUM_PREDICTION_CLASSES) {
                System.out.println("Invalid vector length in test");
                return;
            }

            int bestIndex = 0;
            double bestScore = vector[0];

            for (int j = 1; j < vector.length; j++) {
                if (vector[j] > bestScore) {
                    bestScore = vector[j];
                    bestIndex = j;
                }
            }

            predictions.put(testData.get(datumIndex).word, predictionLabels[bestIndex]);

        }
	}

    private SimpleMatrix forwardProp(List<Datum> data, int datumIndex) {
        SimpleMatrix vectorX = generateWindow(data, windowSize, datumIndex);
        SimpleMatrix vectorZ = W.mult(vectorX);
        SimpleMatrix vectorH = tanh(vectorZ);
        SimpleMatrix vectorV = U.mult(addConstRow(vectorH));
        SimpleMatrix vectorP = softmax(vectorV);

        return vectorP;
    }

    private SimpleMatrix generateWindow(List<Datum> data, int C, int currIndex) {
        SimpleMatrix vectorX = new SimpleMatrix(wordSize * (C + 1), 1);
        int rowIndex = 0;
        for (int i = currIndex - C/2; i <= currIndex + C/2; i++ ) {
            String word = "";
            if (i < 0) {
                word = BEGIN_TOKEN;
            } else if (i >= data.size()) {
                word = END_TOKEN;
            } else {
                word = data.get(i).word;
            }
            double[] wordVector = getWordVector(word);
            for (double w : wordVector) {
                vectorX.set(rowIndex, 0, w);
                rowIndex++;
            }
        }

        // add the constants for the bias
        while (rowIndex < vectorX.numRows()) {
            vectorX.set(rowIndex, 0, 1.0);
            rowIndex++;
        }

        return vectorX;
        
    }

    private double[] getWordVector(String word) {
        int wordIndex = wordToNum.get(FeatureFactory.UNKNOWN_WORD);
        if (wordToNum.containsKey(word)) {
            wordIndex = wordToNum.get(word);
        }

        if (wordIndex < 0 || wordIndex >= L.numCols()) {
            System.out.println("Invalid wordIndex");
            return new double[wordSize];
        }

        double[] vector = L.extractVector(false, wordIndex).getMatrix().getData();
        return vector;

    }

    // add the vector of 1's for the bias
    private SimpleMatrix addConstRow(SimpleMatrix matrix) {
        int numRows = matrix.numRows();
        int numCols = matrix.numCols();
        SimpleMatrix newMatrix = new SimpleMatrix(numRows + 1, numCols);

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                newMatrix.set(i, j, matrix.get(i, j));
            }
        }

        for (int j = 0; j < numCols; j++) {
            newMatrix.set(numRows, j, 1.0);
        }
        return newMatrix;

    }

    private SimpleMatrix tanh(SimpleMatrix matrix) {
        int numRows = matrix.numRows();
        int numCols = matrix.numCols();
        SimpleMatrix newMatrix = new SimpleMatrix(numRows, numCols);

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                newMatrix.set(i, j, Math.tanh(matrix.get(i, j)));
            }
        }
        return newMatrix;
    }


    private SimpleMatrix softmax(SimpleMatrix vectorV) {
        int numRows = vectorV.numRows();
        int numCols = vectorV.numCols();

        if (numCols != 1) {
            System.out.println("Invalid num cols for softmax");
            return vectorV;
        }

        double sum = vectorV.elementSum();
        SimpleMatrix vectorP = new SimpleMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            double val = vectorV.get(i, 0) / sum;
            vectorP.set(i, 0, val);
        }

        return vectorP;
    }
	
}
