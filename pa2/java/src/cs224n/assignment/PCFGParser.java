package cs224n.assignment;

import cs224n.ling.Tree;
import java.util.*;
import cs224n.assignment.*;
import cs224n.ling.Trees;

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
    //        System.out.println("Tree: " + Trees.PennTreeRenderer.render(trainTree));
        }

        lexicon = new Lexicon(binTrees);
        grammar = new Grammar(binTrees);
    }

    private int getInd (String s, Map<String, Integer> aToInd, int[] counter) {
        if (aToInd.containsKey(s)) {
            return aToInd.get(s);
        } else {
            counter[0] += 1;
            aToInd.put(s, counter[0]);
            return counter[0];
        }
    }

    public Tree<String> getBestParse(List<String> sentence) {
        Map<String, Integer> aToInd = new HashMap<String, Integer>();
        int[] counter = {0};
        
        int numWords = sentence.size();
        int aToIndSize = 200;
        double[][][] score = new double[numWords+1][numWords+1][aToIndSize];
        TripletISS[][][] back = new TripletISS[numWords+1][numWords+1][aToIndSize];
        
        for (int i=0; i < numWords; i++) {
            String word = sentence.get(i);
            for (String tag : lexicon.getAllTags()) {
                score[i][i+1][getInd(tag, aToInd, counter)] = lexicon.scoreTagging(word, tag);
                back[i][i+1][getInd(tag, aToInd, counter)] = new TripletISS(-2, word, word);
            }
            // handle unaries
            boolean added = true;
            while (added) {
                added = false;
                Set<String> keySet = new HashSet<String>(aToInd.keySet());
                for (String b : keySet) {
                    int indB = getInd(b, aToInd, counter);
                    if (score[i][i+1][indB] > 0) {
                        List<Grammar.UnaryRule> unaryRuleList = grammar.getUnaryRulesByChild(b);
                        for (Grammar.UnaryRule unaryRule : unaryRuleList) {
                            String a = unaryRule.getParent();
                            int indA = getInd(a, aToInd, counter);
                            double prob = unaryRule.getScore() * score[i][i+1][indB];
                            if (prob > score[i][i+1][indA]) {
                                score[i][i+1][indA] = prob;
                                back[i][i+1][indA] = new TripletISS(-1, b, b);
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
                    Set<String> keySet = new HashSet<String>(aToInd.keySet());
                    for (String b : keySet) {
                        int indB = getInd(b, aToInd, counter);
                        List<Grammar.BinaryRule> binaryRuleList = grammar.getBinaryRulesByLeftChild(b);
                        for (Grammar.BinaryRule binaryRule : binaryRuleList) {
                            String c = binaryRule.getRightChild();
                            String a = binaryRule.getParent();
                            int indC = getInd(c, aToInd, counter);
                            int indA = getInd(a, aToInd, counter);
                            double prob = score[begin][split][indB] * score[split][end][indC] * binaryRule.getScore();
                            if (prob > score[begin][end][indA]){
                                score[begin][end][indA] = prob;
                                back[begin][end][indA] = new TripletISS(split, b, c);
                            }
                        }
                    }
                }
                // handle unaries
                boolean added = true;
                while (added) {
                    added = false;
                    Set<String> keySet = new HashSet<String>(aToInd.keySet());
                    for (String b : keySet) {
                        int indB = getInd(b, aToInd, counter);
                        List<Grammar.UnaryRule> unaryRuleList = grammar.getUnaryRulesByChild(b);
                        for (Grammar.UnaryRule unaryRule : unaryRuleList) {
                            String a = unaryRule.getParent();
                            int indA = getInd(a, aToInd, counter);
                            double prob = unaryRule.getScore() * score[begin][end][indB];
                            if (prob > score[begin][end][indA]) {
                                score[begin][end][indA] = prob;
                                back[begin][end][indA] = new TripletISS(-1, b, b);
                                added = true;
                            }
                        }
                    }
                }
            }
        }
        // rebuild best parse tree
        Tree<String> bestParse = rebuildTree(0, numWords, "ROOT", aToInd, back);
//        System.out.println("Tree: " + Trees.PennTreeRenderer.render(TreeAnnotations.unAnnotateTree(bestParse)));
        // unAnnotate tree        
        return TreeAnnotations.unAnnotateTree(bestParse);
    }

    // rebuild a tree
    private Tree<String> rebuildTree(int begin, int end, String tag, Map<String, Integer> aToInd, TripletISS[][][] back) {
        int[] counter = {0};
        TripletISS backInfo = back[begin][end][getInd(tag, aToInd, counter)];
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
            int indB = getInd(b, aToInd, counter);
            Tree<String> nextNode = rebuildTree (begin, end, b, aToInd, back);
            Tree<String> node = new Tree<String> (tag, Collections.singletonList(nextNode));
            return node;
        }
        //case3: root is nonterm with binary rebuild rule
        else {
            int split =  backInfo.getFirst();
            String leftTag = backInfo.getSecond();
            String rightTag = backInfo.getThird();
            Tree<String> leftNode = rebuildTree(begin, split, leftTag, aToInd, back);
            Tree<String> rightNode = rebuildTree(split, end, rightTag, aToInd, back);
            List<Tree<String>> nextNodeList = new ArrayList<Tree<String>>();
            nextNodeList.add(leftNode);
            nextNodeList.add(rightNode);
            Tree<String> node = new Tree<String>(tag, nextNodeList);
            return node;
        }
    }
}
