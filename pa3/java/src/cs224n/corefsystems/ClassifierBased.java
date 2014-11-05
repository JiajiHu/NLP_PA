package cs224n.corefsystems;

import cs224n.coref.*;
import cs224n.util.Pair;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;
import edu.stanford.nlp.util.logging.StanfordRedwoodConfiguration;

import java.text.DecimalFormat;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * @author Gabor Angeli (angeli at cs.stanford)
 */
public class ClassifierBased implements CoreferenceSystem {

	private static <E> Set<E> mkSet(E[] array){
		Set<E> rtn = new HashSet<E>();
		Collections.addAll(rtn, array);
		return rtn;
	}

	private static final Set<Object> ACTIVE_FEATURES = mkSet(new Object[]{
			Feature.ExactMatch.class,
			
			/*  START: Deemed not useful */
			
			// Feature.MentionDistance.class,
			// Feature.MentionDistanceSentence.class,
			// Feature.FixedIsPronoun.class,
			// Feature.CandidateIsPronoun.class,
			// Feature.ContainsPronoun.class,
			// Feature.FixedEntityType.class,
			// Feature.CandidateEntityType.class,
			// Feature.SameEntityType.class,
			// Pair.make(Feature.MentionDistance.class, Feature.ContainsPronoun.class),
			// Pair.make(Feature.MentionDistanceSentence.class, Feature.ContainsPronoun.class),
			// Pair.make(Feature.FixedEntityType.class, Feature.CandidateEntityType.class),
			
			// Feature.NounPluralCompatible.class,
			
			// Feature.HeadWordSamePOS.class,
			
			/*  END: Deemed not useful */
			
			Pair.make(Feature.MentionDistanceSentence.class, Feature.FixedIsPronoun.class),
			Pair.make(Feature.MentionDistanceSentence.class, Feature.CandidateIsPronoun.class),
			
			Feature.NamePronounGenderIncompatible.class,
			Feature.NamePronounPluralIncompatible.class,
			
			Feature.NounPronounPluralCompatible.class,
			// Feature.NounPronounPluralIncompatible.class,
			
			Feature.Pronoun2GenderCompatible.class,
			Feature.Pronoun2PluralCompatible.class,
			Feature.Pronoun2SpeakerCompatible.class,
			// Feature.Pronoun2GenderIncompatible.class,
			// Feature.Pronoun2PluralIncompatible.class,
			// Feature.Pronoun2SpeakerIncompatible.class,
			Feature.Pronoun2Incompatible.class,

			Feature.HeadWordSame.class,
			// Feature.HeadWordSameLemma.class, // works well, but not as good as HeadWordSame

	});


	private LinearClassifier<Boolean,Feature> classifier;

	public ClassifierBased(){
		StanfordRedwoodConfiguration.setup();
		RedwoodConfiguration.current().collapseApproximate().apply();
	}

	public FeatureExtractor<Pair<Mention,ClusteredMention>,Feature,Boolean> extractor = new FeatureExtractor<Pair<Mention, ClusteredMention>, Feature, Boolean>() {
		private <E> Feature feature(Class<E> clazz, Pair<Mention,ClusteredMention> input, Option<Double> count){
			
			//--Variables
			Mention onPrix = input.getFirst(); //the first mention (referred to as m_i in the handout)
			Mention candidate = input.getSecond().mention; //the second mention (referred to as m_j in the handout)
			Entity candidateCluster = input.getSecond().entity; //the cluster containing the second mention


			//--Features
			if(clazz.equals(Feature.ExactMatch.class)){
				//(exact string match)
				return new Feature.ExactMatch(
					onPrix.gloss().equalsIgnoreCase(candidate.gloss()));

			} else if(clazz.equals(Feature.MentionDistance.class)) {
				// distance between mentions (measured in mentions)
				return new Feature.MentionDistance(
					onPrix.doc.indexOfMention(onPrix) - candidate.doc.indexOfMention(candidate));

			} else if(clazz.equals(Feature.MentionDistanceSentence.class)) {
				// distance between mentions (measured in sentences)
				return new Feature.MentionDistanceSentence(
					onPrix.doc.indexOfSentence(onPrix.sentence) - candidate.doc.indexOfSentence(candidate.sentence));			

			} else if(clazz.equals(Feature.FixedIsPronoun.class)) {
				// whether fixed is pronoun
				return new Feature.FixedIsPronoun(Pronoun.isSomePronoun(onPrix.headWord()));

			} else if(clazz.equals(Feature.CandidateIsPronoun.class)) {
				// whether candidate is pronoun
				return new Feature.CandidateIsPronoun(Pronoun.isSomePronoun(candidate.headWord()));

			} else if(clazz.equals(Feature.ContainsPronoun.class)) {
				// whether either fixed or candidate is pronoun
				return new Feature.ContainsPronoun(
					Pronoun.isSomePronoun(candidate.headWord()) || Pronoun.isSomePronoun(candidate.headWord()));

			} else if(clazz.equals(Feature.FixedEntityType.class)) {
				// NE type of fixed
				return new Feature.FixedEntityType(onPrix.headToken().nerTag());	

			} else if(clazz.equals(Feature.CandidateEntityType.class)) {
				// NE type of candidate
				return new Feature.CandidateEntityType(candidate.headToken().nerTag());	

			} else if(clazz.equals(Feature.SameEntityType.class)) {
				// whether NE type is same between fixed and candidate
				// NOTE: so many "O" tags -- adding special case
				int returnCode = 0;
				if (onPrix.headToken().nerTag().equals("O")) {
					returnCode = (candidate.headToken().nerTag().equals("O")) ? 0 : 1;
				} else if (candidate.headToken().nerTag().equals("O")){
					returnCode = 1;
				} else {
					returnCode = (onPrix.headToken().nerTag().equals(candidate.headToken().nerTag())) ? 3 : 2;
				}
				return new Feature.SameEntityType(returnCode);	

			} else if(clazz.equals(Feature.NamePronounGenderIncompatible.class)) {
				boolean nameProGenderComp = true;
				if (Name.isName(onPrix.headWord()) && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (! Name.mostLikelyGender(onPrix.headWord()).isCompatible(
						Pronoun.valueOrNull(candidate.headWord()).gender)) {
						nameProGenderComp = false;
					}
				} else if (Name.isName(candidate.headWord()) && Pronoun.valueOrNull(onPrix.headWord()) != null) {
					if (! Name.mostLikelyGender(candidate.headWord()).isCompatible(
						Pronoun.valueOrNull(onPrix.headWord()).gender)) {
						nameProGenderComp = false;
					}
				}
				return new Feature.NamePronounGenderIncompatible(nameProGenderComp);

			} else if(clazz.equals(Feature.NamePronounPluralIncompatible.class)) {
				boolean nameProPluralComp = true;
				if (Name.isName(onPrix.headWord()) && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (Pronoun.valueOrNull(candidate.headWord()).plural) {
						nameProPluralComp = false;
					}
				} else if (Name.isName(candidate.headWord()) && Pronoun.valueOrNull(onPrix.headWord()) != null) {
					if (Pronoun.valueOrNull(onPrix.headWord()).plural) {
						nameProPluralComp = false;
					}
				}
				return new Feature.NamePronounPluralIncompatible(nameProPluralComp);
			
			} else if(clazz.equals(Feature.NounPluralCompatible.class)) {
				boolean nounPluralComp = false;
				if (onPrix.headToken().isPluralNoun() && candidate.headToken().isPluralNoun()) {
					nounPluralComp = true;
				}
				return new Feature.NounPluralCompatible(nounPluralComp);

			} else if(clazz.equals(Feature.NounPronounPluralCompatible.class)) {
				boolean nounProPluralComp = false;
				if (onPrix.headToken().isPluralNoun() && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (Pronoun.valueOrNull(candidate.headWord()).plural) {
						nounProPluralComp = true;
					}
				} else if (candidate.headToken().isPluralNoun() && Pronoun.valueOrNull(onPrix.headWord()) != null) {
					if (Pronoun.valueOrNull(onPrix.headWord()).plural) {
						nounProPluralComp = true;
					}
				}
				return new Feature.NounPronounPluralCompatible(nounProPluralComp);

			} else if(clazz.equals(Feature.NounPronounPluralIncompatible.class)) {
				boolean nounProPluralComp = true;
				if (onPrix.headToken().isPluralNoun() && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (!Pronoun.valueOrNull(candidate.headWord()).plural) {
						nounProPluralComp = false;
					}
				} else if (candidate.headToken().isPluralNoun() && Pronoun.valueOrNull(onPrix.headWord()) != null) {
					if (!Pronoun.valueOrNull(onPrix.headWord()).plural) {
						nounProPluralComp = false;
					}
				}
				return new Feature.NounPronounPluralIncompatible(nounProPluralComp);

			} else if(clazz.equals(Feature.Pronoun2GenderCompatible.class)) {
				boolean proGenderComp = false;
				if (Pronoun.valueOrNull(onPrix.headWord()) != null && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (Pronoun.valueOrNull(onPrix.headWord()).gender.isCompatible(
						Pronoun.valueOrNull(candidate.headWord()).gender)) {
						proGenderComp = true;
					}
				}
				return new Feature.Pronoun2GenderCompatible(proGenderComp);

			} else if(clazz.equals(Feature.Pronoun2GenderIncompatible.class)) {
				boolean proGenderComp = true;
				if (Pronoun.valueOrNull(onPrix.headWord()) != null && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (!Pronoun.valueOrNull(onPrix.headWord()).gender.isCompatible(
						Pronoun.valueOrNull(candidate.headWord()).gender)) {
						proGenderComp = false;
					}
				}
				return new Feature.Pronoun2GenderIncompatible(proGenderComp);	

			} else if(clazz.equals(Feature.Pronoun2PluralCompatible.class)) {
				boolean proPluralComp = false;
				if (Pronoun.valueOrNull(onPrix.headWord()) != null && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (Pronoun.valueOrNull(onPrix.headWord()).plural ==
						Pronoun.valueOrNull(candidate.headWord()).plural) {
						proPluralComp = true;
					}
				}
				return new Feature.Pronoun2PluralCompatible(proPluralComp);

			} else if(clazz.equals(Feature.Pronoun2PluralIncompatible.class)) {
				boolean proPluralComp = true;
				if (Pronoun.valueOrNull(onPrix.headWord()) != null && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (Pronoun.valueOrNull(onPrix.headWord()).plural !=
						Pronoun.valueOrNull(candidate.headWord()).plural) {
						proPluralComp = false;
					}
				}
				return new Feature.Pronoun2PluralIncompatible(proPluralComp);

			} else if(clazz.equals(Feature.Pronoun2SpeakerCompatible.class)) {
				boolean proSpeakerComp = false;
				if (Pronoun.valueOrNull(onPrix.headWord()) != null && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (Pronoun.valueOrNull(onPrix.headWord()).speaker ==
						Pronoun.valueOrNull(candidate.headWord()).speaker) {
						proSpeakerComp = true;
					}
				}
				return new Feature.Pronoun2SpeakerCompatible(proSpeakerComp);	

			} else if(clazz.equals(Feature.Pronoun2SpeakerIncompatible.class)) {
				boolean proSpeakerComp = true;
				if (Pronoun.valueOrNull(onPrix.headWord()) != null && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (Pronoun.valueOrNull(onPrix.headWord()).speaker !=
						Pronoun.valueOrNull(candidate.headWord()).speaker) {
						proSpeakerComp = false;
					}
				}
				return new Feature.Pronoun2SpeakerIncompatible(proSpeakerComp);

			} else if(clazz.equals(Feature.Pronoun2Incompatible.class)) {
				boolean proComp = true;
				if (Pronoun.valueOrNull(onPrix.headWord()) != null && Pronoun.valueOrNull(candidate.headWord()) != null) {
					if (!Pronoun.valueOrNull(onPrix.headWord()).gender.isCompatible(
						Pronoun.valueOrNull(candidate.headWord()).gender)) {
						proComp = false;
					} else if (Pronoun.valueOrNull(onPrix.headWord()).plural !=
						Pronoun.valueOrNull(candidate.headWord()).plural) {
						proComp = false;
					} else if (Pronoun.valueOrNull(onPrix.headWord()).speaker !=
						Pronoun.valueOrNull(candidate.headWord()).speaker) {
						proComp = false;
					}
				}
				return new Feature.Pronoun2Incompatible(proComp);

			} else if(clazz.equals(Feature.HeadWordSame.class)) {
				return new Feature.HeadWordSame(onPrix.headWord().equalsIgnoreCase(candidate.headWord()));
			
			} else if(clazz.equals(Feature.HeadWordSameLemma.class)) {
				return new Feature.HeadWordSameLemma(onPrix.headToken().lemma().equalsIgnoreCase(candidate.headToken().lemma()));

			} else if(clazz.equals(Feature.HeadWordSamePOS.class)) {
				return new Feature.HeadWordSamePOS(onPrix.headToken().posTag().equals(candidate.headToken().posTag()));
			
			} else {
				throw new IllegalArgumentException("Unregistered feature: " + clazz);
			}
		}

		@SuppressWarnings({"unchecked"})
		@Override
		protected void fillFeatures(Pair<Mention, ClusteredMention> input, Counter<Feature> inFeatures, Boolean output, Counter<Feature> outFeatures) {
			//--Input Features
			for(Object o : ACTIVE_FEATURES){
				if(o instanceof Class){
					//(case: singleton feature)
					Option<Double> count = new Option<Double>(1.0);
					Feature feat = feature((Class) o, input, count);
					if(count.get() > 0.0){
						inFeatures.incrementCount(feat, count.get());
					}
				} else if(o instanceof Pair){
					//(case: pair of features)
					Pair<Class,Class> pair = (Pair<Class,Class>) o;
					Option<Double> countA = new Option<Double>(1.0);
					Option<Double> countB = new Option<Double>(1.0);
					Feature featA = feature(pair.getFirst(), input, countA);
					Feature featB = feature(pair.getSecond(), input, countB);
					if(countA.get() * countB.get() > 0.0){
						inFeatures.incrementCount(new Feature.PairFeature(featA, featB), countA.get() * countB.get());
					}
				}
			}

			//--Output Features
			if(output != null){
				outFeatures.incrementCount(new Feature.CoreferentIndicator(output), 1.0);
			}
		}

		@Override
		protected Feature concat(Feature a, Feature b) {
			return new Feature.PairFeature(a,b);
		}
	};

	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		startTrack("Training");
		//--Variables
		RVFDataset<Boolean, Feature> dataset = new RVFDataset<Boolean, Feature>();
		LinearClassifierFactory<Boolean, Feature> fact = new LinearClassifierFactory<Boolean,Feature>();
		//--Feature Extraction
		startTrack("Feature Extraction");
		for(Pair<Document,List<Entity>> datum : trainingData){
			//(document variables)
			Document doc = datum.getFirst();
			List<Entity> goldClusters = datum.getSecond();
			List<Mention> mentions = doc.getMentions();
			Map<Mention,Entity> goldEntities = Entity.mentionToEntityMap(goldClusters);
			startTrack("Document " + doc.id);
			//(for each mention...)
			for(int i=0; i<mentions.size(); i++){
				//(get the mention and its cluster)
				Mention onPrix = mentions.get(i);
				Entity source = goldEntities.get(onPrix);
				if(source == null){ throw new IllegalArgumentException("Mention has no gold entity: " + onPrix); }
				//(for each previous mention...)
				int oldSize = dataset.size();
				for(int j=i-1; j>=0; j--){
					//(get previous mention and its cluster)
					Mention cand = mentions.get(j);
					Entity target = goldEntities.get(cand);
					if(target == null){ throw new IllegalArgumentException("Mention has no gold entity: " + cand); }
					//(extract features)
					Counter<Feature> feats = extractor.extractFeatures(Pair.make(onPrix, cand.markCoreferent(target)));
					//(add datum)
					dataset.add(new RVFDatum<Boolean, Feature>(feats, target == source));
					//(stop if
					if(target == source){ break; }
				}
				//logf("Mention %s (%d datums)", onPrix.toString(), dataset.size() - oldSize);
			}
			endTrack("Document " + doc.id);
		}
		endTrack("Feature Extraction");
		//--Train Classifier
		startTrack("Minimizer");
		this.classifier = fact.trainClassifier(dataset);
		endTrack("Minimizer");
		//--Dump Weights
		startTrack("Features");
		//(get labels to print)
		Set<Boolean> labels = new HashSet<Boolean>();
		labels.add(true);
		//(print features)
		for(Triple<Feature,Boolean,Double> featureInfo : this.classifier.getTopFeatures(labels, 0.0, true, 100, true)){
			Feature feature = featureInfo.first();
			Boolean label = featureInfo.second();
			Double magnitude = featureInfo.third();
			//log(FORCE,new DecimalFormat("0.000").format(magnitude) + " [" + label + "] " + feature);
		}
		end_Track("Features");
		endTrack("Training");
	}

	public List<ClusteredMention> runCoreference(Document doc) {
		//--Overhead
		startTrack("Testing " + doc.id);
		//(variables)
		List<ClusteredMention> rtn = new ArrayList<ClusteredMention>(doc.getMentions().size());
		List<Mention> mentions = doc.getMentions();
		int singletons = 0;
		//--Run Classifier
		for(int i=0; i<mentions.size(); i++){
			//(variables)
			Mention onPrix = mentions.get(i);
			int coreferentWith = -1;
			//(get mention it is coreferent with)
			for(int j=i-1; j>=0; j--){
				ClusteredMention cand = rtn.get(j);
				boolean coreferent = classifier.classOf(new RVFDatum<Boolean, Feature>(extractor.extractFeatures(Pair.make(onPrix, cand))));
				if(coreferent){
					coreferentWith = j;
					break;
				}
			}
			//(mark coreference)
			if(coreferentWith < 0){
				singletons += 1;
				rtn.add(onPrix.markSingleton());
			} else {
				//log("Mention " + onPrix + " coreferent with " + mentions.get(coreferentWith));
				rtn.add(onPrix.markCoreferent(rtn.get(coreferentWith)));
			}
		}
		//log("" + singletons + " singletons");
		//--Return
		endTrack("Testing " + doc.id);
		return rtn;
	}

	private class Option<T> {
		private T obj;
		public Option(T obj){ this.obj = obj; }
		public Option(){};
		public T get(){ return obj; }
		public void set(T obj){ this.obj = obj; }
		public boolean exists(){ return obj != null; }
	}
}
