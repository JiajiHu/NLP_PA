package cs224n.assignment;

import cs224n.ling.Tree;
import cs224n.util.*;
import java.util.*;

/**
 * The CKY PCFG Parser you will implement.
 */
public class PCFGParser implements Parser {
    private Grammar grammar;
    private Lexicon lexicon;
    private Interner<Triplet<Integer, Integer, String>> canonT;
    private Interner<Pair<Integer, Integer>> canonP;
    
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

    public Tree<String> getBestParse(List<String> sentence) {
        long startTime = System.currentTimeMillis();
        int numWords = sentence.size();
        IdentityCounterMap<Pair<Integer, Integer>, String> score = new IdentityCounterMap<Pair<Integer, Integer>, String>();
        Map<Triplet<Integer, Integer, String>, Triplet<Integer, String, String>> back = 
                new IdentityHashMap<Triplet<Integer, Integer, String>, Triplet<Integer, String, String>>();
        canonT = new Interner<Triplet<Integer, Integer, String>>();
        canonP = new Interner<Pair<Integer, Integer>>();
        
        for (int i=0; i < numWords; i++) {
            String word = sentence.get(i);
            Pair<Integer, Integer> point = canonP.intern(new Pair(i, i+1));
            for (String tag : lexicon.getAllTags()) {
                score.setCount(point, tag, lexicon.scoreTagging(word, tag));
                back.put(canonT.intern(new Triplet(i, i+1 ,tag)), new Triplet(-2, word, word));
            }
            // handle unaries
            boolean added = true;
            while (added) {
                added = false;
                Set<String> keySet = new HashSet<String>(score.getCounter(point).keySet());
                for (String b : keySet) {
                    double bScore = score.getCount(point, b);
                    if (bScore > 0) {
                        List<Grammar.UnaryRule> unaryRuleList = grammar.getUnaryRulesByChild(b);
                        for (Grammar.UnaryRule unaryRule : unaryRuleList) {
                            String a = unaryRule.getParent();
                            double prob = unaryRule.getScore() * bScore;
                            if (prob > score.getCount(point, a)) {
                                score.setCount(point, a, prob);
                                back.put(canonT.intern(new Triplet(i, i+1 ,a)), new Triplet(-1, b, b));
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
                Pair<Integer, Integer> pointA = canonP.intern(new Pair(begin, end));
                for (int split=begin+1; split < end; split++) {
                    Pair<Integer, Integer> pointB = canonP.intern(new Pair(begin, split));
                    Pair<Integer, Integer> pointC = canonP.intern(new Pair(split, end));
                    Set<String> keySet = new HashSet<String>(score.getCounter(pointB).keySet());
                    for (String b : keySet) {
                        List<Grammar.BinaryRule> binaryRuleList = grammar.getBinaryRulesByLeftChild(b);
                        for (Grammar.BinaryRule binaryRule : binaryRuleList) {
                            String c = binaryRule.getRightChild();
                            String a = binaryRule.getParent();
                            double prob = score.getCount(pointB, b) * score.getCount(pointC, c) * binaryRule.getScore();
                            if (prob > score.getCount(pointA, a)){
                                score.setCount(pointA, a, prob);
                                back.put(canonT.intern(new Triplet(begin, end ,a)), new Triplet(split, b, c));
                            }
                        }
                    }
                }
                // handle unaries
                boolean added = true;
                while (added) {
                    added = false;
                    Set<String> keySet = new HashSet<String>(score.getCounter(pointA).keySet());
                    for (String b : keySet) {
                        List<Grammar.UnaryRule> unaryRuleList = grammar.getUnaryRulesByChild(b);
                        for (Grammar.UnaryRule unaryRule : unaryRuleList) {
                            String a = unaryRule.getParent();
                            double prob = unaryRule.getScore() * score.getCount(pointA, b);
                            if (prob > score.getCount(pointA, a)) {
                                score.setCount(pointA, a, prob);
                                back.put(canonT.intern(new Triplet(begin, end ,a)), new Triplet(-1, b, b));
                                added = true;
                            }
                        }
                    }
                }
            }
        }
        // rebuild best parse tree
        Tree<String> bestParse = rebuildTree(0, numWords, "ROOT", back);
        System.out.println("Time elapsed: " + (System.currentTimeMillis() - startTime)/1000.0);
        return TreeAnnotations.unAnnotateTree(bestParse);
    }

    // rebuild a tree
    private Tree<String> rebuildTree(int begin, int end, String tag, Map<Triplet<Integer, Integer, String>, Triplet<Integer, String, String>> back) {
        Triplet<Integer, String, String> backInfo = back.get(canonT.intern(new Triplet(begin, end, tag)));
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
