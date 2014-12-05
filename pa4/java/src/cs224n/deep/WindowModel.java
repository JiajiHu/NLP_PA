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

	public int windowSize, wordSize, hiddenSize, iterations;
    public double learningRate;

    public String[] predictionLabels = new String[]{"O", "LOC", "MISC", "ORG", "PER"};
    public Map<String, Integer> labelToIndex = new HashMap();
    public static int NUM_PREDICTION_CLASSES = 5;


    public static final String BEGIN_TOKEN = "<s>";
    public static final String END_TOKEN = "</s>";

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
        L = FeatureFactory.allVecs;
        wordToNum = FeatureFactory.wordToNum;
        predictions = new HashMap<String, String>();

        iterations = 30;
		windowSize = _windowSize;
        hiddenSize = _hiddenSize;
        learningRate = _lr;
        wordSize = L.numRows();

        int labelIndex = 0;
        for (String s : predictionLabels) {
            labelToIndex.put(s, labelIndex++);
        }
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
        this.W = SimpleMatrix.random(hiddenSize, wordSize * windowSize  + 1, -epsilon, epsilon, new Random());
        this.U = SimpleMatrix.random(NUM_PREDICTION_CLASSES, hiddenSize + 1, -epsilon, epsilon, new Random());

	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> trainData){
        for (int i = 0; i< iterations; i++) {
            System.out.println("Iteration: " + i);
            for (int datumIndex = 0; datumIndex < trainData.size(); datumIndex++) {
                
                List<Integer> wordNums = generateWordNumList(trainData, windowSize, datumIndex);
                SimpleMatrix vectorX = generateWindow(trainData, windowSize, datumIndex);
                SimpleMatrix vectorZ = W.mult(vectorX);
                SimpleMatrix vectorH = tanh(vectorZ);
                SimpleMatrix vectorV = U.mult(addConstRow(vectorH));
                SimpleMatrix vectorP = softmax(vectorV);

                // System.out.println();
                // System.out.println("U: " + U.numRows() + ", " + U.numCols());
                // System.out.println("W: " + W.numRows() + ", " + W.numCols());
                // System.out.println("L: " + L.numRows() + ", " + L.numCols());
                // System.out.println();
                // System.out.println("X: " + vectorX.numRows() + ", " + vectorX.numCols());
                // System.out.println("Z: " + vectorZ.numRows() + ", " + vectorZ.numCols());
                // System.out.println("H: " + vectorH.numRows() + ", " + vectorH.numCols());
                // System.out.println("V: " + vectorV.numRows() + ", " + vectorV.numCols());
                // System.out.println("P: " + vectorP.numRows() + ", " + vectorP.numCols());

                int labelNum = labelToIndex.get(trainData.get(datumIndex).label);
                List<SimpleMatrix> deltas = getDeltas(labelNum, vectorH, vectorP);
                List<SimpleMatrix> gradients = getGradients(false, vectorX, vectorH, deltas);
                
                //check = gradCheck(regOn, vec, W, U, y, grads);
                
                oneSGD(gradients, wordNums);
            }
        }
	}

    private List<SimpleMatrix> getDeltas(int labelNum, SimpleMatrix vectorH, SimpleMatrix vectorP){
        List<SimpleMatrix> deltas = new ArrayList<SimpleMatrix>();
        
        SimpleMatrix delta2 = new SimpleMatrix(NUM_PREDICTION_CLASSES,1);
        for (int i = 0; i<NUM_PREDICTION_CLASSES; i++) {
            delta2.set(i, 0, vectorP.get(i,0) - (i == labelNum ? 1 : 0));
        }
        
        SimpleMatrix delta1 = elementWiseMultMat(removeConstRow(U.transpose().mult(delta2)), elementWiseTanhGrad(vectorH));    
        SimpleMatrix delta0 = W.transpose().mult(delta1);    
        // System.out.println("delta0: " + delta0.numRows() + ", " + delta0.numCols());
        // System.out.println("delta1: " + delta1.numRows() + ", " + delta1.numCols());
        // System.out.println("delta2: " + delta2.numRows() + ", " + delta2.numCols());
        deltas.add(delta0);
        deltas.add(delta1);
        deltas.add(delta2); //put them in intuitive order!
        
        return deltas;
    }

    private List<SimpleMatrix> getGradients(boolean hasRegularization, SimpleMatrix vectorX, SimpleMatrix vectorH, List<SimpleMatrix> deltas){
        List<SimpleMatrix> gradients = new ArrayList<SimpleMatrix>();
        if (hasRegularization) {
            // TODO: add regularization
        } else {
            gradients.add(removeConstRow(deltas.get(0))); //partial{J}{x}
            SimpleMatrix partialW = new SimpleMatrix(W.numRows(), W.numCols());
            partialW.insertIntoThis(0, 0, deltas.get(1).mult(vectorX.transpose()));//partial{J}{W}
            partialW.insertIntoThis(0, W.numRows()-1, deltas.get(1));//partial{J}{b1}
            gradients.add(partialW);
            SimpleMatrix partialU = new SimpleMatrix(U.numRows(), U.numCols());
            partialU.insertIntoThis(0, 0, deltas.get(2).mult(vectorH.transpose())); //partial{J}{U}
            partialU.insertIntoThis(0, U.numRows()-1, deltas.get(2)); //partial{J}{b2}
            gradients.add(partialU);
        }

        // System.out.println("partialx: " + gradients.get(0).numRows() + ", " + gradients.get(0).numCols());
        // System.out.println("partialW: " + gradients.get(1).numRows() + ", " + gradients.get(1).numCols());
        // System.out.println("partialU: " + gradients.get(2).numRows() + ", " + gradients.get(2).numCols());

        return gradients;
    }

    private void oneSGD(List<SimpleMatrix> gradients, List<Integer> wordNums){

        for (int i=0; i<windowSize; i++){
            SimpleMatrix oldL = L.extractMatrix(0, L.numRows(), wordNums.get(i), wordNums.get(i)+1);
            SimpleMatrix newL = oldL.plus(elementWiseMultScalar(gradients.get(0).extractMatrix(i*wordSize, i*wordSize+wordSize, 0, 1), -learningRate));
            L.insertIntoThis(0, wordNums.get(i), newL);
        }
        W = W.plus(elementWiseMultScalar(gradients.get(1), -learningRate));
        U = U.plus(elementWiseMultScalar(gradients.get(2), -learningRate));
    }

    private SimpleMatrix removeConstRow(SimpleMatrix m){
        return m.extractMatrix(0, m.numRows()-1, 0, m.numCols());
    }
	
    private SimpleMatrix appendCol(SimpleMatrix m1, SimpleMatrix m2){
        m1.insertIntoThis(0, m1.numCols()-1, m2);
        return m1;
    }

    private SimpleMatrix elementWiseTanhGrad(SimpleMatrix m){
        SimpleMatrix res = new SimpleMatrix(m.numRows(), m.numCols());
        for (int row = 0; row < m.numRows(); row++){
            for (int col = 0; col < m.numCols(); col++){
                res.set(row, col, 1 - m.get(row, col) * m.get(row, col));
            }
        }
        return res;
    }
    private SimpleMatrix elementWiseMultMat(SimpleMatrix m1, SimpleMatrix m2){
        SimpleMatrix res = new SimpleMatrix(m1.numRows(), m1.numCols());
        for (int row = 0; row < res.numRows(); row++){
            for (int col = 0; col < res.numCols(); col++){
                res.set(row, col, m1.get(row, col) * m2.get(row, col));
            }
        }
        return res;
    }
    private SimpleMatrix elementWiseMultScalar(SimpleMatrix m, double s){
        SimpleMatrix res = new SimpleMatrix(m.numRows(), m.numCols());
        for (int row = 0; row < m.numRows(); row ++){
            for (int col = 0; col < m.numCols(); col ++){
                res.set(row, col, m.get(row, col) * s);
            }
        }
        return res;
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

    private List<Integer> generateWordNumList(List<Datum> data, int C, int currIndex){
        List<Integer> wordNumList = new ArrayList();
        for (int i = currIndex - C/2; i <= currIndex + C/2; i++ ) {
            String word = "";
            if (i < 0) {
                word = BEGIN_TOKEN;
            } else if (i >= data.size()) {
                word = END_TOKEN;
            } else {
                word = data.get(i).word;
            }
            int wordNum = wordToNum.get(FeatureFactory.UNKNOWN_WORD);
            if (wordToNum.containsKey(word)){
                wordNum = wordToNum.get(word);
            }
            wordNumList.add(wordNum);
        }
        return wordNumList;
    } 

    private SimpleMatrix generateWindow(List<Datum> data, int C, int currIndex) {
        SimpleMatrix vectorX = new SimpleMatrix(wordSize * C + 1, 1);
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
        vectorX.set(rowIndex, 0, 1.0);
        rowIndex++;

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

        double sum = 0;
        for (int i = 0; i < numRows; i++){
            sum += Math.exp(vectorV.get(i, 0));
        }

        SimpleMatrix vectorP = new SimpleMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            double val = Math.exp(vectorV.get(i, 0)) / sum;
            vectorP.set(i, 0, val);
        }

        return vectorP;
    }
	
    
}
