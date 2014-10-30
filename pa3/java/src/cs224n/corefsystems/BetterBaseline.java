package cs224n.corefsystems;

import java.util.*;


import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.*;
import cs224n.util.Pair;

public class BetterBaseline implements CoreferenceSystem {

    private Set<Pair<String, String>> headWordPairSet;

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		headWordPairSet = new HashSet<Pair<String, String>>();
        for(Pair<Document, List<Entity>> pair : trainingData){
            List<Entity> clusters = pair.getSecond();

            for (Entity cluster : clusters) {
                for(Pair<Mention, Mention> mentionPair : cluster.orderedMentionPairs()) {
                    Mention m1 = mentionPair.getFirst();
                    Mention m2 = mentionPair.getSecond();
                    String headWord1 = m1.headWord();
                    String headWord2 = m2.headWord();
                    Pair<String, String> headWordPair = new Pair<String, String>(headWord1, headWord2);
                    headWordPairSet.add(headWordPair);
                }

            }

        }

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
        List<ClusteredMention> clusteredMentions = new ArrayList<ClusteredMention>();
        Map<String, Entity> headWordClusterMap = new HashMap<String, Entity>();
        for(Mention m : doc.getMentions()){
            String headWord = m.headWord();
            if (headWordClusterMap.containsKey(headWord)) {
                Entity cluster = headWordClusterMap.get(headWord);
                clusteredMentions.add(m.markCoreferent(cluster));
            } else {
                Entity cluster = new Entity(new ArrayList<Mention>(), m);
                clusteredMentions.add(m.markCoreferent(cluster));
                headWordClusterMap.put(headWord, cluster);
            }

        }
        return clusteredMentions;
    }

}
