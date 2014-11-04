package cs224n.coref;

import cs224n.util.Pair;

import java.util.Set;

/**
 * @author Gabor Angeli (angeli at cs.stanford)
 */
public interface Feature {

  //-----------------------------------------------------------
  // TEMPLATE FEATURE TEMPLATES
  //-----------------------------------------------------------
  public static class PairFeature implements Feature {
    public final Pair<Feature,Feature> content;
    public PairFeature(Feature a, Feature b){ this.content = Pair.make(a, b); }
    public String toString(){ return content.toString(); }
    public boolean equals(Object o){ return o instanceof PairFeature && ((PairFeature) o).content.equals(content); }
    public int hashCode(){ return content.hashCode(); }
  }

  public static abstract class Indicator implements Feature {
    public final boolean value;
    public Indicator(boolean value){ this.value = value; }
    public boolean equals(Object o){ return o instanceof Indicator && o.getClass().equals(this.getClass()) && ((Indicator) o).value == value; }
    public int hashCode(){ 
    	return this.getClass().hashCode() ^ Boolean.valueOf(value).hashCode(); }
    public String toString(){ 
    	return this.getClass().getSimpleName() + "(" + value + ")"; }
  }

  public static abstract class IntIndicator implements Feature {
    public final int value;
    public IntIndicator(int value){ this.value = value; }
    public boolean equals(Object o){ return o instanceof IntIndicator && o.getClass().equals(this.getClass()) && ((IntIndicator) o).value == value; }
    public int hashCode(){ 
    	return this.getClass().hashCode() ^ value; 
    }
    public String toString(){ return this.getClass().getSimpleName() + "(" + value + ")"; }
  }

  public static abstract class BucketIndicator implements Feature {
    public final int bucket;
    public final int numBuckets;
    public BucketIndicator(int value, int max, int numBuckets){
      this.numBuckets = numBuckets;
      bucket = value * numBuckets / max;
      if(bucket < 0 || bucket >= numBuckets){ throw new IllegalStateException("Bucket out of range: " + value + " max="+max+" numbuckets="+numBuckets); }
    }
    public boolean equals(Object o){ return o instanceof BucketIndicator && o.getClass().equals(this.getClass()) && ((BucketIndicator) o).bucket == bucket; }
    public int hashCode(){ return this.getClass().hashCode() ^ bucket; }
    public String toString(){ return this.getClass().getSimpleName() + "(" + bucket + "/" + numBuckets + ")"; }
  }

  public static abstract class Placeholder implements Feature {
    public Placeholder(){ }
    public boolean equals(Object o){ return o instanceof Placeholder && o.getClass().equals(this.getClass()); }
    public int hashCode(){ return this.getClass().hashCode(); }
    public String toString(){ return this.getClass().getSimpleName(); }
  }

  public static abstract class StringIndicator implements Feature {
    public final String str;
    public StringIndicator(String str){ this.str = str; }
    public boolean equals(Object o){ return o instanceof StringIndicator && o.getClass().equals(this.getClass()) && ((StringIndicator) o).str.equals(this.str); }
    public int hashCode(){ return this.getClass().hashCode() ^ str.hashCode(); }
    public String toString(){ return this.getClass().getSimpleName() + "(" + str + ")"; }
  }

  public static abstract class SetIndicator implements Feature {
    public final Set<String> set;
    public SetIndicator(Set<String> set){ this.set = set; }
    public boolean equals(Object o){ return o instanceof SetIndicator && o.getClass().equals(this.getClass()) && ((SetIndicator) o).set.equals(this.set); }
    public int hashCode(){ return this.getClass().hashCode() ^ set.hashCode(); }
    public String toString(){
      StringBuilder b = new StringBuilder();
      b.append(this.getClass().getSimpleName());
      b.append("( ");
      for(String s : set){
        b.append(s).append(" ");
      }
      b.append(")");
      return b.toString();
    }
  }
  
  /*
   * TODO: If necessary, add new feature types
   */

  //-----------------------------------------------------------
  // REAL FEATURE TEMPLATES
  //-----------------------------------------------------------

  public static class CoreferentIndicator extends Indicator {
    public CoreferentIndicator(boolean coreferent){ super(coreferent); }
  }

  public static class ExactMatch extends Indicator {
    public ExactMatch(boolean exactMatch){ super(exactMatch); }
  }
  
  public static class MentionDistance extends IntIndicator {
    public MentionDistance(int mentionDist){ super(mentionDist); }
  }

  public static class MentionDistanceSentence extends IntIndicator {
    public MentionDistanceSentence(int mentionDistSen){ super(mentionDistSen); }
  }
  
  public static class FixedIsPronoun extends Indicator {
    public FixedIsPronoun(boolean isPronoun){ super(isPronoun); }
  }
  
  public static class CandidateIsPronoun extends Indicator {
    public CandidateIsPronoun(boolean isPronoun){ super(isPronoun); }
  }
  
  public static class ContainsPronoun extends Indicator {
    public ContainsPronoun(boolean containsPronoun){ super(containsPronoun); }
  }
  
  public static class FixedEntityType extends StringIndicator {
    public FixedEntityType(String fixedEntity){ super(fixedEntity); }
  }

  public static class CandidateEntityType extends StringIndicator {
    public CandidateEntityType(String candEntity){ super(candEntity); }
  }  

  public static class SameEntityType extends IntIndicator {
    public SameEntityType(int code){ super(code); }
  }
  
  public static class NamePronounGenderIncompatible extends Indicator {
    public NamePronounGenderIncompatible(boolean compatible){ super(compatible); }
  }

  public static class NamePronounPluralIncompatible extends Indicator {
    public NamePronounPluralIncompatible(boolean compatible){ super(compatible); }
  }

  public static class NounPronounPluralCompatible extends Indicator {
    public NounPronounPluralCompatible(boolean compatible){ super(compatible); }
  }

  public static class NounPronounPluralIncompatible extends Indicator {
    public NounPronounPluralIncompatible(boolean compatible){ super(compatible); }
  }

  public static class NounPluralCompatible extends Indicator {
    public NounPluralCompatible(boolean compatible){ super(compatible); }
  }

  public static class Pronoun2GenderCompatible extends Indicator {
    public Pronoun2GenderCompatible(boolean compatible){ super(compatible); }
  }

  public static class Pronoun2GenderIncompatible extends Indicator {
    public Pronoun2GenderIncompatible(boolean compatible){ super(compatible); }
  }

  public static class Pronoun2PluralCompatible extends Indicator {
    public Pronoun2PluralCompatible(boolean compatible){ super(compatible); }
  }
  
  public static class Pronoun2PluralIncompatible extends Indicator {
    public Pronoun2PluralIncompatible(boolean compatible){ super(compatible); }
  }

  public static class Pronoun2SpeakerCompatible extends Indicator {
    public Pronoun2SpeakerCompatible(boolean compatible){ super(compatible); }
  }

  public static class Pronoun2SpeakerIncompatible extends Indicator {
    public Pronoun2SpeakerIncompatible(boolean compatible){ super(compatible); }
  }

  public static class Pronoun2Incompatible extends Indicator {
    public Pronoun2Incompatible(boolean compatible){ super(compatible); }
  }

  public static class HeadWordSame extends Indicator {
    public HeadWordSame(boolean same){ super(same); }
  }

  public static class HeadWordSameLemma extends Indicator {
    public HeadWordSameLemma(boolean same){ super(same); }
  }

  public static class HeadWordSamePOS extends Indicator {
    public HeadWordSamePOS(boolean same){ super(same); }
  }
}
