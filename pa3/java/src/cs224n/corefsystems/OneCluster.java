package cs224n.corefsystems;

import java.util.*;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.coref.Mention;
import cs224n.util.Pair;


public class OneCluster implements CoreferenceSystem {

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// No need for training in this baseline system
        return;

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
        List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();
        Entity cluster = new Entity(new ArrayList<Mention>());
        for(Mention m : doc.getMentions()){
            mentions.add(m.markCoreferent(cluster));
        }
        return mentions;
	}

}
