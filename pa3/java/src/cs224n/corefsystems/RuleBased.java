package cs224n.corefsystems;

import java.util.*;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.coref.Mention;
import cs224n.util.Pair;

public class RuleBased implements CoreferenceSystem {

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// No training for rule-based system
        return;

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

        // Exact String Match
        for (Set<Mention> group1 : mentionGroups) {
            for (Set<Mention> group2 : mentionGroups) {
                if (!group1.equals(group2)) {
                    boolean shouldMerge = exactStringMatch(group1, group2);
                    if (shouldMerge) {
                        group1.addAll(group2);
                        group2.removeAll(group2);
                    }
                }
            }
        }

        // Head Word Match
        for (Set<Mention> group1 : mentionGroups) {
            for (Set<Mention> group2 : mentionGroups) {
                if (!group1.equals(group2)) {
                    boolean shouldMerge = headWordMatch(group1, group2);
                    if (shouldMerge) {
                        group1.addAll(group2);
                        group2.removeAll(group2);
                    }
                }
            }
        }

        // Head Token Lemma Match
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


    private boolean exactStringMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            for (Mention m2 : group2) {
                if (m1.gloss().equalsIgnoreCase(m2.gloss())) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean headWordMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            for (Mention m2 : group2) {
                if (m1.headWord().equalsIgnoreCase(m2.headWord())) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean headTokenLemmaMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            for (Mention m2 : group2) {
                if (m1.headToken().lemma().equalsIgnoreCase(m2.headToken().lemma())) {
                    return true;
                }
            }
        }
        return false;
    }


}
