package cs224n.assignment;

import cs224n.ling.Tree;
import cs224n.util.Triplet;
import java.util.*;

/**
 * The CKY PCFG Parser you will implement.
 */
public class PCFGParser implements Parser {
    private Grammar grammar;
    private Lexicon lexicon;

    public void train(List<Tree<String>> trainTrees) {
        // Before we generate our grammar, the training trees
        // need to be binarized so that rules are at most binary
        List<Tree<String>> binTrees = new ArrayList<Tree<String>>();
        for (Tree<String> trainTree: trainTrees) {
            binTrees.add(TreeAnnotations.annotateTree(trainTree));
        }

        lexicon = new Lexicon(binTrees);
        grammar = new Grammar(binTrees);
    }

    // private int getInd (String s, Map<String, Integer> aToInd, int[] counter) {
    //     if (aToInd.containsKey(s)) {
    //         return aToInd.get(s);
    //     } else {
    //         counter[0] += 1;
    //         aToInd.put(s, counter[0]);
    //         return counter[0];
    //     }
    // }

    public Tree<String> getBestParse(List<String> sentence) {
        Set<String> tagDict = new HashSet<String>();
        
        int numWords = sentence.size();
        Map<Triplet<Integer, Integer, String>, Double> score = new HashMap<Triplet<Integer, Integer, String>, Double>();
        Map<Triplet<Integer, Integer, String>, Triplet<Integer, String, String>> back = new HashMap<Triplet<Integer, Integer, String>, Triplet<Integer, String, String>>();
        
        for (int i=0; i < numWords; i++) {
            String word = sentence.get(i);
            for (String tag : lexicon.getAllTags()) {
                tagDict.add(tag);
                Triplet<Integer, Integer, String> point = new Triplet<Integer, Integer, String>(i, i+1, tag);
                score.put(point, lexicon.scoreTagging(word, tag));
                back.put(point, new Triplet<Integer, String, String>(-2, word, word));
            }
            // handle unaries
            boolean added = true;
            while (added) {
                added = false;
                Set<String> keySet = new HashSet<String>(tagDict);
                for (String b : keySet) {
                    Triplet<Integer, Integer, String> pointB = new Triplet<Integer, Integer, String>(i, i+1, b);
                    double bScore = score.containsKey(pointB) ? score.get(pointB) : 0.0;
                    if (bScore > 0) {
                        List<Grammar.UnaryRule> unaryRuleList = grammar.getUnaryRulesByChild(b);
                        for (Grammar.UnaryRule unaryRule : unaryRuleList) {
                            String a = unaryRule.getParent();
                            tagDict.add(a);
                            Triplet<Integer, Integer, String> pointA = new Triplet<Integer, Integer, String>(i, i+1, a);
                            double prob = unaryRule.getScore() * bScore;
                            if (!score.containsKey(pointA) || prob > score.get(pointA)) {
                                score.put(pointA, prob);
                                back.put(pointA, new Triplet<Integer, String, String>(-1, b, b));
                                added = true;
                            }
                        }
                    }
                }            
            }
        }

        
        for (int span=2; span < numWords + 1; span++ ) {
            for (int begin=0; begin < numWords +1 - span; begin++) {
                int end = begin + span;
                for (int split=begin+1; split < end; split++) {
                    Set<String> keySet = new HashSet<String>(tagDict);
                    for (String b : keySet) {
                        Triplet<Integer, Integer, String> pointB = new Triplet<Integer, Integer, String>(begin, split, b);
                        List<Grammar.BinaryRule> binaryRuleList = grammar.getBinaryRulesByLeftChild(b);
                        for (Grammar.BinaryRule binaryRule : binaryRuleList) {
                            String c = binaryRule.getRightChild();
                            String a = binaryRule.getParent();
                            tagDict.add(a);
                            Triplet<Integer, Integer, String> pointC = new Triplet<Integer, Integer, String>(split, end, c);
                            Triplet<Integer, Integer, String> pointA = new Triplet<Integer, Integer, String>(begin, end, a);
                            double scoreB = score.containsKey(pointB) ? score.get(pointB) : 0;
                            double scoreC = score.containsKey(pointC) ? score.get(pointC) : 0;
                            double prob = scoreB * scoreC * binaryRule.getScore();
                            if (!score.containsKey(pointA) || prob > score.get(pointA)){
                                score.put(pointA, prob);
                                back.put(pointA, new Triplet<Integer, String, String>(split, b, c));
                            }
                        }
                    }
                }
                // handle unaries
                boolean added = true;
                while (added) {
                    added = false;
                    Set<String> keySet = new HashSet<String>(tagDict);
                    for (String b : keySet) {
                        Triplet<Integer, Integer, String> pointB = new Triplet<Integer, Integer, String>(begin, end, b);
                        List<Grammar.UnaryRule> unaryRuleList = grammar.getUnaryRulesByChild(b);
                        for (Grammar.UnaryRule unaryRule : unaryRuleList) {
                            String a = unaryRule.getParent();
                            tagDict.add(a);
                            Triplet<Integer, Integer, String> pointA = new Triplet<Integer, Integer, String>(begin, end, a);
                            double scoreB = score.containsKey(pointB) ? score.get(pointB) : 0;
                            double prob = unaryRule.getScore() * scoreB;
                            if (!score.containsKey(pointA) || prob > score.get(pointA)) {
                                score.put(pointA, prob);
                                back.put(pointA, new Triplet<Integer, String, String>(-1, b, b));
                                added = true;
                            }
                        }
                    }
                }
            }
        }
        // rebuild best parse tree
        Tree<String> bestParse = rebuildTree(0, numWords, "ROOT", back);
//        System.out.println("Tree: " + Trees.PennTreeRenderer.render(TreeAnnotations.unAnnotateTree(bestParse)));
        // unAnnotate tree        
        return TreeAnnotations.unAnnotateTree(bestParse);
    }

    // rebuild a tree
    private Tree<String> rebuildTree(int begin, int end, String tag, Map<Triplet<Integer, Integer, String>, Triplet<Integer, String, String>> back) {
        Triplet<Integer, String, String> backInfo = back.get(new Triplet<Integer, Integer, String>(begin, end, tag));
        if (backInfo == null) {
            return new Tree<String>(tag); 
        }
        // case1: root is preterminal
        if (backInfo.getFirst() == -2) {
            String b = backInfo.getSecond();
            Tree<String> nextNode = new Tree<String> (b);
            Tree<String> node = new Tree<String> (tag, Collections.singletonList(nextNode));
            return node;
        }
        // case2: root is nonterm with unary rebuild rule
        if (backInfo.getFirst() == -1) {
            String b = backInfo.getSecond();
            Tree<String> nextNode = rebuildTree (begin, end, b, back);
            Tree<String> node = new Tree<String> (tag, Collections.singletonList(nextNode));
            return node;
        }
        //case3: root is nonterm with binary rebuild rule
        else {
            int split =  backInfo.getFirst();
            String leftTag = backInfo.getSecond();
            String rightTag = backInfo.getThird();
            Tree<String> leftNode = rebuildTree(begin, split, leftTag, back);
            Tree<String> rightNode = rebuildTree(split, end, rightTag, back);
            List<Tree<String>> nextNodeList = new ArrayList<Tree<String>>();
            nextNodeList.add(leftNode);
            nextNodeList.add(rightNode);
            Tree<String> node = new Tree<String>(tag, nextNodeList);
            return node;
        }
    }
}
