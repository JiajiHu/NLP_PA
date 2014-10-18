package cs224n.assignment;

/**
 * A tiplet of Integer, String, String
 * @author Jiaji Hu, modifying code by Paul Baumstarck
 */
public class TripletISS {
	Integer first;
	String second;
	String third;
	
	public Integer getFirst() {
		return first;
	}
	
	public String getSecond() {
		return second;
	}
	
	public String getThird() {
		return third;
	}
	
	public boolean equals(Object o) {
		if (this == o) return true;
		if (!(o instanceof TripletISS)) return false;
	
		@SuppressWarnings("unchecked")	
		final TripletISS triplet = (TripletISS) o;
		
		if (first != null ? !first.equals(triplet.first) : triplet.first != null) return false;
		if (second != null ? !second.equals(triplet.second) : triplet.second != null) return false;
		if (third != null ? !third.equals(triplet.third) : triplet.third != null) return false;
		
		return true;
	}
	
	public int hashCode() {
		int result;
		result = (first != null ? first.hashCode() : 0);
		result = 29 * result + (second != null ? second.hashCode() : 0);
		result = 37 * result + (second != null ? second.hashCode() : 0);
		return result;
	}
	
	public String toString() {
		return "(" + getFirst() + ", " + getSecond() + ", " + getThird() + ")";
	}
	
	public TripletISS(Integer first, String second, String third) {
		this.first = first;
		this.second = second;
		this.third = third;
	}
}
