package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {

    private static void writeResults(String outputFileName, List<Datum> testData, Map<String, String> predictions) {
        Writer writer = null;
        try {
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFileName), "utf-8"));
            for (Datum d : testData) {
                String word = d.word;
                String label = d.label;

                String prediction = "O";
                if (predictions != null && predictions.containsKey(word)) {
                    prediction = predictions.get(word);

                }
                writer.write(word + "\t" + prefixedLabel(label) + "\t" + prefixedLabel(prediction) + "\n");
            }
        } catch (IOException ex) {
            System.out.println("Output writer exception " + ex);
        } finally {
            try {writer.close();} catch (Exception ex) {}
        }

    }

    private static String prefixedLabel(String label) {
        String prefixedLabel = label;
        if (!label.isEmpty() && !label.equals("O")) {
            prefixedLabel = "I-" + label;
        }
        return prefixedLabel;
    }
    
    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev ../output/output1 (NUM_ITERS)");
            return;
        }

        // this reads in the train and test datasets
        List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
        List<Datum> testData = FeatureFactory.readTestData(args[1]);

        // read output file name
        String outputFileName = args[2];

        //	read the train and test data
        FeatureFactory.initializeVocab("../data/vocab_glove.txt");

        // load word vectors from file
        SimpleMatrix allVecs= FeatureFactory.readWordVectors("../data/wordVectors_glove.txt");

        // randomly initialize word vectors
        // SimpleMatrix allVecs = FeatureFactory.randomInitializeWordVectors(50, FeatureFactory.wordToNum.size());


        // initialize model
        int iterations = 20;
        if (args.length > 3) {
            iterations = Integer.parseInt(args[3]);
        }

    	// Baseline model
        // BaselineModel model = new BaselineModel();
        
        WindowModel model = new WindowModel(3, 20, 0.001, false);
    	model.initWeights();

        model.test(testData); //test result
        Map<String, String> predictions= model.getPredictions();
        writeResults(outputFileName+"_dev_iter0", testData, predictions);
        
        model.test(trainData); //also get train F1
        predictions= model.getPredictions();
        writeResults(outputFileName+"_train_iter0", trainData, predictions);

        for (int i = 1; i<= iterations; i++) {
            System.out.println("Iteration: " + i);
        
            model.train(trainData);
            model.test(testData); //test result
            predictions= model.getPredictions();
            writeResults(outputFileName+"_dev_iter"+i, testData, predictions);
            
            model.test(trainData); //also get train F1
            predictions= model.getPredictions();
            writeResults(outputFileName+"_train_iter"+i, trainData, predictions);
        }

    }
}