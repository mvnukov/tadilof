package knn.hnsw;

import lombok.ToString;
import lombok.extern.java.Log;

import java.util.Comparator;

@ToString(exclude = "distanceComparator")
@Log
public class NodeIdAndDistance<Distance> implements Comparable<NodeIdAndDistance<Distance>> {

    final int nodeId;
    final Distance distance;
    final Comparator<Distance> distanceComparator;

    NodeIdAndDistance(int nodeId, Distance distance, Comparator<Distance> distanceComparator) {
        this.nodeId = nodeId;
        this.distance = distance;
        this.distanceComparator = distanceComparator;
    }

    @Override
    public int compareTo(NodeIdAndDistance<Distance> o) {
        return distanceComparator.compare(distance, o.distance);
    }
}
