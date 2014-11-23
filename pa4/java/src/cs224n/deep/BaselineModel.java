package cs224n.deep;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by zhangtong on 11/22/14.
 */
public class BaselineModel {
    public static final String UNKNOWN_TAG = "O";
    private HashMap<String, String> predictions;
    private HashMap<String, String> wordLabelMap;

    public BaselineModel() {
        predictions = new HashMap<String, String>();
        wordLabelMap = new HashMap<String, String>();
    }

    public HashMap<String, String> getPredictions() {
        return predictions;
    }

    public void train(List<Datum> trainData ){
        for (Datum d : trainData) {
            String word = d.word;
            String label = d.label;
            if (wordLabelMap.containsKey(word)) {
//                if (!wordLabelMap.get(word).equals(label)) {
//                    wordLabelMap.put(word, UNKNOWN_TAG);
//                }
            } else {
                wordLabelMap.put(word, label);
            }
        }
    }


    public void test(List<Datum> testData){
        for (Datum d : testData) {
            String word = d.word;
            String label = "";
            if (wordLabelMap.containsKey(word)) {
                label = wordLabelMap.get(word);
            } else {
                label = UNKNOWN_TAG;
            }
            predictions.put(word, label);
        }
    }
}
