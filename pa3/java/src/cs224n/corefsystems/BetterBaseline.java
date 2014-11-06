package cs224n.corefsystems;

import java.util.*;


import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.*;
import cs224n.util.Pair;

public class BetterBaseline implements CoreferenceSystem {

    private Set<Pair<String, String>> headWordPairs = new HashSet<Pair<String, String>>();
    private Set<Pair<String, String>> headTokenLemmaPairs = new HashSet<Pair<String, String>>();

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
                    String lemma1 = m1.headToken().lemma();
                    String lemma2 = m2.headToken().lemma();

                    headWordPairs.add(new Pair<String, String>(hw1, hw2));
                    headWordPairs.add(new Pair<String, String>(hw2, hw1));

                    headTokenLemmaPairs.add(new Pair<String, String>(lemma1, lemma2));
                    headTokenLemmaPairs.add(new Pair<String, String>(lemma2, lemma1));

                }

            }

        }

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
        List<ClusteredMention> clusteredMentions = new ArrayList<ClusteredMention>();

        Set<Set<Mention>> mentionGroups = new HashSet<Set<Mention>>();

        // Initialize each mention to its own group
        for (Mention m : doc.getMentions()) {
            Set<Mention> singleMentionGroup = new HashSet<Mention>();
            singleMentionGroup.add(m);
            mentionGroups.add(singleMentionGroup);
        }

        // Head Word Match
        for (Set<Mention> group1 : mentionGroups) {
            for (Set<Mention> group2 : mentionGroups) {
                if (!group1.equals(group2)) {
                    boolean shouldMerge = headTokenLemmaMatch(group1, group2);
                    if (shouldMerge) {
                        group1.addAll(group2);
                        group2.removeAll(group2);
                    }
                }
            }
        }

        for (Set<Mention> group : mentionGroups) {
            List<Mention> mentionList = new ArrayList<Mention>();
            mentionList.addAll(group);
            Entity cluster = new Entity(mentionList);
            for (Mention m : group) {
                ClusteredMention clusteredMention = m.markCoreferent(cluster);
                clusteredMentions.add(clusteredMention);
            }
        }


        return clusteredMentions;
    }

    private boolean headWordMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            for (Mention m2 : group2) {
                String hw1 = m1.headWord();
                String hw2 = m2.headWord();
                if (headWordPairs.contains(new Pair<String, String>(hw1, hw2))) {
                    return true;
                }

            }
        }
        return false;
    }

    private boolean headTokenLemmaMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            for (Mention m2 : group2) {
                String lemma1 = m1.headToken().lemma();
                String lemma2 = m2.headToken().lemma();
                if (headTokenLemmaPairs.contains(new Pair<String, String>(lemma1, lemma2))) {
                    return true;
                }

            }
        }
        return false;
    }

}
