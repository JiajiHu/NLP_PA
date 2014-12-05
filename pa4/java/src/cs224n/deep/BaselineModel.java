package cs224n.deep;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by zhangtong on 11/22/14.
 */
public class BaselineModel {
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
        wordLabelMap.put(FeatureFactory.UNKNOWN_WORD, "O");
        for (Datum d : trainData) {
            String word = d.word.toLowerCase();
            String label = d.label;
            if (wordLabelMap.containsKey(word)) {
                if (!wordLabelMap.get(word).equals(label)) {
                    if (label != "O") {
                        wordLabelMap.put(word, label);
                    }
//                    wordLabelMap.put(word, "O");
                }
            } else {
                wordLabelMap.put(word, label);
            }
        }
    }


    public void test(List<Datum> testData){
        for (Datum d : testData) {
            String word = d.word.toLowerCase();
            String label = wordLabelMap.get(FeatureFactory.UNKNOWN_WORD);
            if (wordLabelMap.containsKey(word)) {
                label = wordLabelMap.get(word);
            }
            predictions.put(word, label);
        }
    }
}
