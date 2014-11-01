package cs224n.corefsystems;

import java.util.*;


import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.*;
import cs224n.util.Pair;

public class BetterBaseline implements CoreferenceSystem {

    private Set<Set<String>> headWordGroups = new HashSet<Set<String>>();
    private Set<String> possibleHeadWords = new HashSet<String>();

    private Set<String> findHeadWordGroup(String headWord) {
        for (Set<String> group : headWordGroups) {
            if (group.contains(headWord)) {
                return group;
            }
        }
        return null;
    }

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {

        for(Pair<Document, List<Entity>> pair : trainingData){
            List<Entity> clusters = pair.getSecond();

            for (Entity cluster : clusters) {
                for(Pair<Mention, Mention> mentionPair : cluster.orderedMentionPairs()) {
                    Mention m1 = mentionPair.getFirst();
                    Mention m2 = mentionPair.getSecond();
                    String hw1 = m1.headWord();
                    String hw2 = m2.headWord();

                    if (!possibleHeadWords.contains(hw1) && !possibleHeadWords.contains(hw2)) {
                        Set<String> newSet = new HashSet<String>();
                        newSet.add(hw1);
                        newSet.add(hw2);
                        headWordGroups.add(newSet);
                        possibleHeadWords.addAll(newSet);
                    } else if (!possibleHeadWords.contains(hw1)) {
                        Set<String> group = findHeadWordGroup(hw2);
                        if (group != null) {
                            headWordGroups.remove(group);
                            group.add(hw1);
                            headWordGroups.add(group);
                            possibleHeadWords.add(hw1);
                        }
                    } else if (!possibleHeadWords.contains(hw2)) {
                        Set<String> group = findHeadWordGroup(hw1);
                        if (group != null) {
                            headWordGroups.remove(group);
                            group.add(hw2);
                            headWordGroups.add(group);
                            possibleHeadWords.add(hw2);
                        }
                    } else {
                        Set<String> group1 = findHeadWordGroup(hw1);
                        Set<String> group2 = findHeadWordGroup(hw2);

                        if (group1 != null && group2 != null) {
                            headWordGroups.remove(group1);
                            headWordGroups.remove(group2);
                            group1.addAll(group2);
                            headWordGroups.add(group1);
                        } else if (group1 != null) {
                            headWordGroups.remove(group1);
                            group1.add(hw2);
                            headWordGroups.add(group1);
                        } else if (group2 != null) {
                            headWordGroups.remove(group2);
                            group2.add(hw1);
                            headWordGroups.add(group2);
                        }
                    }

                }

            }

        }

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
        List<ClusteredMention> clusteredMentions = new ArrayList<ClusteredMention>();
        Map<Set<String>, Entity> headWordsClusterMap = new HashMap<Set<String>, Entity>();

        for (Mention m : doc.getMentions()) {
            String headWord = m.headWord();
            if (possibleHeadWords.contains(headWord)) {
                Set<String> group = findHeadWordGroup(headWord);
                if (group != null) {
                    if (headWordsClusterMap.containsKey(group)) {
                        Entity cluster = headWordsClusterMap.get(group);
                        clusteredMentions.add(m.markCoreferent(cluster));
                    } else {
                        Entity cluster = new Entity(new ArrayList<Mention>(), m);
                        clusteredMentions.add(m.markCoreferent(cluster));
                        headWordsClusterMap.put(group, cluster);
                    }
                } else {
                    System.out.println("Group == null!");
                }

            } else {
                Set<String> group = new HashSet<String>();
                group.add(headWord);
                Entity cluster = new Entity(new ArrayList<Mention>(), m);
                clusteredMentions.add(m.markCoreferent(cluster));
                headWordsClusterMap.put(group, cluster);
            }
        }

        return clusteredMentions;
    }

}
