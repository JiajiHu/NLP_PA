package cs224n.corefsystems;

import java.util.*;

import cs224n.coref.*;
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
        Set<Set<Mention>> pronounMentionGroups = new HashSet<Set<Mention>>();

        // Initialize each mention to its own group
        for (Mention m : doc.getMentions()) {
            Set<Mention> singleMentionGroup = new HashSet<Mention>();
            singleMentionGroup.add(m);
            if (Pronoun.valueOrNull(m.headWord()) != null) {
                pronounMentionGroups.add(singleMentionGroup);
            } else {
                String nerTag = m.headToken().nerTag();

                mentionGroups.add(singleMentionGroup);
            }
//            System.out.println("NER Tag: " + m.headWord() + " " + m.headToken().nerTag());
//            System.out.println("Mention Parse: " + m.parse);
        }

        // Pronoun Match
        for (Set<Mention> group1 : pronounMentionGroups) {
            for (Set<Mention> group2 : pronounMentionGroups) {
                if (!group1.equals(group2)) {
                    boolean shouldMerge = pronounMatch(group1, group2);
                    if (shouldMerge) {
                        group1.addAll(group2);
                        group2.removeAll(group2);
                    }
                }
            }
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

        // Head Token Noun Match
        for (Set<Mention> group1 : mentionGroups) {
            for (Set<Mention> group2 : mentionGroups) {
                if (!group1.equals(group2)) {
                    boolean shouldMerge = mentionParseMatch(group1, group2);
                    if (shouldMerge) {
                        group1.addAll(group2);
                        group2.removeAll(group2);
                    }
                }
            }
        }

        for (Set<Mention> group : pronounMentionGroups) {
            List<Mention> mentionList = new ArrayList<Mention>();
            mentionList.addAll(group);
            Entity cluster = new Entity(mentionList);
            for (Mention m : group) {
                ClusteredMention clusteredMention = m.markCoreferent(cluster);
                clusteredMentions.add(clusteredMention);
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

    private boolean pronounMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            Pronoun pronoun1 = Pronoun.valueOrNull(m1.headWord());
            if (pronoun1 == null) {
                break;
            }
            for (Mention m2 : group2) {
                Pronoun pronoun2 = Pronoun.valueOrNull(m2.headWord());
                if (pronoun2 == null) {
                    break;
                }
                if (pronoun1.gender == pronoun2.gender
                        && pronoun1.speaker == pronoun2.speaker
                        && pronoun1.plural == pronoun2.plural) {
                    if (pronoun1.speaker != Pronoun.Speaker.THIRD_PERSON) {
                        return true;
                    } else {
                        int index1 = m1.doc.indexOfSentence(m1.sentence);
                        int index2 = m2.doc.indexOfSentence(m2.sentence);
                        if (Math.abs(index1 - index2) <= 1) {
                            return true;
                        }
                    }

                }
            }
        }
        return false;
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


    private boolean headTokenPluralNounMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            for (Mention m2 : group2) {
                if (m1.headToken().isPluralNoun() && m2.headToken().isPluralNoun()) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean headTokenSpeakerMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            for (Mention m2 : group2) {
                if (m1.headToken().speaker().equals(m2.headToken().speaker())) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean headTokenNERMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            for (Mention m2 : group2) {
                if (m1.headToken().nerTag().equals(m2.headToken().nerTag())) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean mentionParseMatch(Set<Mention> group1, Set<Mention> group2) {
        for (Mention m1 : group1) {
            for (Mention m2 : group2) {
                if (m1.parse.equals(m2.parse)) {
                    return true;
                }
            }
        }
        return false;
    }


}
