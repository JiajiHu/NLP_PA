package cs224n.util;
import java.util.*;

public class IdentityTripletMap<K, V> {
	IdentityHashMap<K,HashMap<V, Triplet>> data = new IdentityHashMap<K,HashMap<V, Triplet>>();
	
	public Set<K> keySet() {
		return data.keySet();
	}
	
	public void put(K k, V v, Triplet tri) {
		HashMap<V, Triplet> thisData = getData(k);
		thisData.put(v, tri);
	}
	
	public Triplet get(K k, V v) {
		HashMap<V, Triplet> thisData = getData(k);
		return thisData.get(v);
	}
	
	private HashMap<V, Triplet> getData(K k) {
		HashMap<V, Triplet> vToObj = data.get(k);
		if (vToObj == null) {
			vToObj = new HashMap<V, Triplet>();
			data.put(k, vToObj);
		}
		return vToObj;
	}
	
}
