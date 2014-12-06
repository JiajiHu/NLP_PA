package cs224n.deep;
import java.lang.*;
import java.security.cert.CollectionCertStoreParameters;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, U, W2;

    private HashMap<String, Integer> wordToNum;
    private HashMap<String, String> predictions;

	public int windowSize, wordSize, hiddenSize, hiddenSize2;
    public double learningRate, lambda;
    public boolean hasRegularization, useSequenceModel, deeperLearning;
    private int additionalDim;

    public String[] predictionLabels = new String[]{"O", "LOC", "MISC", "ORG", "PER"};
    public Map<String, Integer> labelToIndex = new HashMap();
    public static int NUM_PREDICTION_CLASSES = 5;

    private double[] previousPredictions = null;

    public static final String BEGIN_TOKEN = "<s>";
    public static final String END_TOKEN = "</s>";

	public WindowModel(int _windowSize, int _hiddenSize, double _lr, boolean _hasReg, double _lambda, boolean _useSeqModel, boolean _deeper, int _hiddenSize2){
        L = FeatureFactory.allVecs;
        wordToNum = FeatureFactory.wordToNum;
        predictions = new HashMap<String, String>();

        hasRegularization = _hasReg;
		windowSize = _windowSize;
        hiddenSize = _hiddenSize;
        hiddenSize2 = _hiddenSize2;
        learningRate = _lr;
        lambda = _lambda;
        wordSize = L.numRows();

        useSequenceModel = _useSeqModel;
        // if use sequence model, increase the dimension by K (label classes)
        additionalDim = useSequenceModel ? predictionLabels.length : 0;


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
        this.W = SimpleMatrix.random(hiddenSize, wordSize * windowSize  + 1 + additionalDim, -epsilon, epsilon, new Random());
        if (deeperLearning) {
            this.W2 = SimpleMatrix.random(hiddenSize2, hiddenSize + 1, -epsilon, epsilon, new Random());
            this.U = SimpleMatrix.random(NUM_PREDICTION_CLASSES, hiddenSize2 + 1, -epsilon, epsilon, new Random());
        } else {
            this.U = SimpleMatrix.random(NUM_PREDICTION_CLASSES, hiddenSize + 1, -epsilon, epsilon, new Random());
        }
    }


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> trainData){
        previousPredictions = null;


        // generate random permutation for training data to improve SGD efficiency
        List<Integer> trainingIndices = new ArrayList<Integer>();
        for (int i = 0; i < trainData.size(); i++) {
            trainingIndices.add(i);
        }
        Collections.shuffle(trainingIndices);

        for (int datumIndex : trainingIndices) {
            List<SimpleMatrix> vectors = new ArrayList<SimpleMatrix>();

            List<Integer> wordNums = generateWordNumList(trainData, windowSize, datumIndex);
            SimpleMatrix vectorX = generateWindow(trainData, windowSize, datumIndex);
            vectors.add(vectorX);
            SimpleMatrix vectorZ = W.mult(vectorX);
            vectors.add(vectorZ);
            SimpleMatrix vectorH = tanh(vectorZ);
            vectors.add(vectorH);
            SimpleMatrix vectorV = null;
            if (deeperLearning) {
                SimpleMatrix vectorVmid = W2.mult(addConstRow(vectorH));
                vectors.add(vectorVmid);
                SimpleMatrix vectorHmid = tanh(vectorVmid);
                vectors.add(vectorHmid);
                vectorV = U.mult(addConstRow(vectorHmid));
            } else {
                vectorV = U.mult(addConstRow(vectorH));
            }
            vectors.add(vectorV);
            SimpleMatrix vectorP = softmax(vectorV);
            vectors.add(vectorP);
            
            int labelNum = labelToIndex.get(trainData.get(datumIndex).label);
            List<SimpleMatrix> deltas = getDeltas(labelNum, vectors);
            List<SimpleMatrix> gradients = getGradients(vectors, deltas);
            
            // gradientCheck(labelNum, vectorX, gradients, true, true, true);
            
            oneSGD(gradients, wordNums);

            // previousPredictions = new double[additionalDim];
            // previousPredictions[labelNum] = 1;

            previousPredictions = vectorP.extractVector(false, 0).getMatrix().getData();

        }
	}

    private List<SimpleMatrix> getDeltas(int labelNum, List<SimpleMatrix> vectors){
        List<SimpleMatrix> deltas = new ArrayList<SimpleMatrix>();
        SimpleMatrix vectorP = vectors.get(vectors.size()-1);

        SimpleMatrix delta2 = new SimpleMatrix(NUM_PREDICTION_CLASSES,1);
        for (int i = 0; i<NUM_PREDICTION_CLASSES; i++) {
            delta2.set(i, 0, vectorP.get(i,0) - (i == labelNum ? 1 : 0));
        }
        SimpleMatrix delta1mid = null;
        SimpleMatrix delta1 = null;
        SimpleMatrix delta0 = null;
        if (deeperLearning) {
            delta1mid = elementWiseMultMat(removeConstRow(U.transpose().mult(delta2)), elementWiseTanhGrad(vectors.get(4)));    
            delta1 = elementWiseMultMat(removeConstRow(W2.transpose().mult(delta1mid)), elementWiseTanhGrad(vectors.get(2)));    
            delta0 = W.transpose().mult(delta1);    
        } else {
            delta1 = elementWiseMultMat(removeConstRow(U.transpose().mult(delta2)), elementWiseTanhGrad(vectors.get(2)));    
            delta0 = W.transpose().mult(delta1);    
        }
        deltas.add(delta0);
        deltas.add(delta1);
        if (deeperLearning){
            deltas.add(delta1mid);
        }
        deltas.add(delta2); //put them in intuitive order!
        return deltas;
    }

    private List<SimpleMatrix> getGradients(List<SimpleMatrix> vectors, List<SimpleMatrix> deltas){
        List<SimpleMatrix> gradients = new ArrayList<SimpleMatrix>();
        SimpleMatrix vectorX = vectors.get(0);
        gradients.add(removeConstRow(deltas.get(0))); //partial{J}{x}
        SimpleMatrix partialW = new SimpleMatrix(W.numRows(), W.numCols());
        partialW.insertIntoThis(0, 0, deltas.get(1).mult(vectorX.transpose()));//partial{J}{W}
        partialW.insertIntoThis(0, W.numCols()-1, deltas.get(1));//partial{J}{b1}
        
        SimpleMatrix partialW2 = null;
        SimpleMatrix partialU = null;

        if (deeperLearning) {
            partialW2 = new SimpleMatrix(W2.numRows(), W2.numCols());
            partialW2.insertIntoThis(0, 0, deltas.get(3).mult(vectors.get(4).transpose())); //partial{J}{U}
            partialW2.insertIntoThis(0, W2.numCols()-1, deltas.get(3)); //partial{J}{b2}

            partialU = new SimpleMatrix(U.numRows(), U.numCols());
            partialU.insertIntoThis(0, 0, deltas.get(2).mult(vectors.get(2).transpose())); //partial{J}{U}
            partialU.insertIntoThis(0, U.numCols()-1, deltas.get(2)); //partial{J}{b2}

        } else {
            partialU = new SimpleMatrix(U.numRows(), U.numCols());
            partialU.insertIntoThis(0, 0, deltas.get(2).mult(vectors.get(2).transpose())); //partial{J}{U}
            partialU.insertIntoThis(0, U.numCols()-1, deltas.get(2)); //partial{J}{b2}
        }

        if (hasRegularization) {
            partialW = partialW.plus(elementWiseMultScalar(W, lambda));
            if (deeperLearning) {
                partialW2 = partialW2.plus(elementWiseMultScalar(W2, lambda));
            }
            partialU = partialU.plus(elementWiseMultScalar(U, lambda));
        }

        gradients.add(partialW);
        if (deeperLearning) {
            gradients.add(partialW2);
        }
        gradients.add(partialU);
        return gradients;
    }

    private double costFunction(int labelNum, SimpleMatrix vectorP) {
        // excluding division of training size, because it's just a constant
        double regCost = 0.0;
        if (hasRegularization){
            SimpleMatrix newW = setConstColToZero(W);
            SimpleMatrix newU = setConstColToZero(U);
            regCost = (elementWiseMultMat(newW, newW).elementSum() + elementWiseMultMat(newU, newU).elementSum()) * lambda / 2;
            if (deeperLearning){
                SimpleMatrix newW2 = setConstColToZero(W2);
                regCost += elementWiseMultMat(newW2, newW2).elementSum() * lambda / 2
            }
        }
        return -Math.log(vectorP.get(labelNum, 0)) + regCost;
    }

    private void gradientCheck(int labelNum, SimpleMatrix vectorX, List<SimpleMatrix> gradients, 
                               boolean checkX, boolean checkW, boolean checkU) {
        /* For the gradient check, we ran over the whole X, W and U to make sure it was correct.
        In this version, we are skipping by 10 or 20 to make it run fast.
        We also ran it on small windows (as suggested), though that case is already covered by the full run.*/
        double epsilon = 1e-4;
        double maxAbs = 1e-7;
        int checkWindow = 10;
        int checkStart = 0;
        /* check partial_x */
        if (checkX) {
            SimpleMatrix partialX = gradients.get(0);
            SimpleMatrix numericalPartialX = new SimpleMatrix(partialX.numRows(), partialX.numCols());
            SimpleMatrix maskX = new SimpleMatrix(vectorX.numRows(), vectorX.numCols());
            for (int i=0; i<partialX.numRows(); i+=20){ //skipping by 20 to save time. Also ran with i++ for one whole iteration
                maskX.set(i, 0, epsilon);
                double cost1 = costFunction(labelNum, forwardProp(vectorX.plus(maskX)));
                maskX.set(i, 0, -epsilon);
                double cost2 = costFunction(labelNum, forwardProp(vectorX.plus(maskX)));
                maskX.set(i, 0, 0.0);

                numericalPartialX.set(i, 0, (cost1 - cost2)/(2.0 * epsilon));

                // System.out.println(partialX.get(i,0) + ", " + numericalPartialX.get(i,0));
                if (Math.abs(partialX.get(i,0) - numericalPartialX.get(i,0)) > maxAbs) {
                    System.out.println("Gradient check failed at partial X. Problem at position: " + i);
                }
            }
        }
        /* check partial_W */
        if (checkW) {
            SimpleMatrix partialW = gradients.get(1);
            SimpleMatrix numericalPartialW = new SimpleMatrix(partialW.numRows(), partialW.numCols());
            SimpleMatrix maskW = new SimpleMatrix(W.numRows(), W.numCols());
            for (int i=0; i<partialW.numRows(); i+=20){ //skipping by 20 to save time. Also ran with i++ for one whole iteration
                for (int j=0; j<partialW.numCols(); j+=20){ //skipping by 20 to save time. Also ran with j++ for one whole iteration
                    maskW.set(i, j, epsilon);
                    W = W.plus(maskW);
                    double cost1 = costFunction(labelNum, forwardProp(vectorX));
                    maskW.set(i, j, -2*epsilon);
                    W = W.plus(maskW);
                    double cost2 = costFunction(labelNum, forwardProp(vectorX));
                    maskW.set(i, j, epsilon);
                    W = W.plus(maskW);
                    maskW.set(i, j, 0.0);
                    numericalPartialW.set(i, j, (cost1 - cost2)/(2.0 * epsilon));
                    
                    // System.out.println(partialW.get(i,j) + ", " + numericalPartialW.get(i,j));
                    if (Math.abs(partialW.get(i,j) - numericalPartialW.get(i,j)) > maxAbs) {
                        System.out.println("Gradient check failed at partial W. Problem at position: " + i + ", " + j);
                        System.out.println(partialW.get(i,j) + ", " + numericalPartialW.get(i,j));
                    }
                }
            }
            if (deeperLearning) { // check W2
                // TODO
            }
        }
        if (checkU) {
            if (deeperLearning) {
                SimpleMatrix partialU = gradients.get(3);
            } else {
                SimpleMatrix partialU = gradients.get(2);
            }
            SimpleMatrix numericalPartialU = new SimpleMatrix(partialU.numRows(), partialU.numCols());
            SimpleMatrix maskU = new SimpleMatrix(U.numRows(), U.numCols());
            for (int i=0; i<partialU.numRows(); i+=10){ //skipping by 10 to save time. Also ran with i++ for one whole iteration
                for (int j=0; j<partialU.numCols(); j+=10){ //skipping by 10 to save time. Also ran with j++ for one whole iteration
                    maskU.set(i, j, epsilon);
                    U = U.plus(maskU);
                    double cost1 = costFunction(labelNum, forwardProp(vectorX));
                    maskU.set(i, j, -2*epsilon);
                    U = U.plus(maskU);
                    double cost2 = costFunction(labelNum, forwardProp(vectorX));
                    maskU.set(i, j, epsilon);
                    U = U.plus(maskU);
                    maskU.set(i, j, 0.0);
                    numericalPartialU.set(i, j, (cost1 - cost2)/(2.0 * epsilon));
                    
                    // System.out.println(partialU.get(i,j) + ", " + numericalPartialU.get(i,j));
                    if (Math.abs(partialU.get(i,j) - numericalPartialU.get(i,j)) > maxAbs) {
                        System.out.println("Gradient check failed at partial U. Problem at position: " + i + ", " + j);
                        System.out.println(partialU.get(i,j) + ", " + numericalPartialU.get(i,j));
                    }
                }
            }
        }
    }

    private void oneSGD(List<SimpleMatrix> gradients, List<Integer> wordNums){

        for (int i=0; i<windowSize; i++){
            SimpleMatrix oldL = L.extractMatrix(0, L.numRows(), wordNums.get(i), wordNums.get(i)+1);
            SimpleMatrix newL = oldL.plus(elementWiseMultScalar(gradients.get(0).extractMatrix(i*wordSize, i*wordSize+wordSize, 0, 1), -learningRate));
            L.insertIntoThis(0, wordNums.get(i), newL);
        }
        W = W.plus(elementWiseMultScalar(gradients.get(1), -learningRate));
        if (deeperLearning) {
            W2 = W2.plus(elementWiseMultScalar(gradients.get(2), -learningRate));
            U = U.plus(elementWiseMultScalar(gradients.get(3), -learningRate));
        } else {
            U = U.plus(elementWiseMultScalar(gradients.get(2), -learningRate));
        }
    }

    private SimpleMatrix removeConstRow(SimpleMatrix m){
        return m.extractMatrix(0, m.numRows()-1, 0, m.numCols());
    }

    private SimpleMatrix setConstColToZero(SimpleMatrix m){
        SimpleMatrix res = new SimpleMatrix(m.numRows(), m.numCols());
        for (int i=0; i<m.numRows(); i++){
            res.set(i, m.numCols()-1, 0.1234);
        }
        return res;
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
        previousPredictions = null;
        for (int datumIndex = 0; datumIndex < testData.size(); datumIndex++) {
            // forward propagation
            SimpleMatrix vectorX = generateWindow(testData, windowSize, datumIndex);
            SimpleMatrix vectorP = forwardProp(vectorX);
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
            
            previousPredictions = vector;

        }
	}

    private SimpleMatrix forwardProp(SimpleMatrix vectorX) {
        SimpleMatrix vectorZ = W.mult(vectorX);
        SimpleMatrix vectorH = tanh(vectorZ);
        SimpleMatrix vectorV = null;
        if (deeperLearning) {
            SimpleMatrix vectorVmid = W2.mult(addConstRow(vectorH));
            SimpleMatrix vectorHmid = tanh(vectorVmid);
            vectorV = U.mult(addConstRow(vectorHmid));
        } else {
            vectorV = U.mult(addConstRow(vectorH));
        }
        SimpleMatrix vectorP = softmax(vectorV);

        return vectorP;
    }

    public static String replaceDigits(String string) {
        return string.replaceAll("\\d", FeatureFactory.DIGIT_WORD);
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
                word = replaceDigits(data.get(i).word.toLowerCase());
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
        SimpleMatrix vectorX = new SimpleMatrix(wordSize * C + 1 + additionalDim, 1);
        int rowIndex = 0;
        for (int i = currIndex - C/2; i <= currIndex + C/2; i++ ) {
            String word = "";
            if (i < 0) {
                word = BEGIN_TOKEN;
            } else if (i >= data.size()) {
                word = END_TOKEN;
            } else {
                word = replaceDigits(data.get(i).word.toLowerCase());
            }
            double[] wordVector = getWordVector(word);
            for (double w : wordVector) {
                vectorX.set(rowIndex, 0, w);
                rowIndex++;
            }
        }

        // add the previous predictions
        if (useSequenceModel) {
            for (int i = 0; i < additionalDim; i++) {
                double val = 0.0;
                if (previousPredictions != null) {
                    val = previousPredictions[i];
                }
                vectorX.set(rowIndex, 0, val);
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
