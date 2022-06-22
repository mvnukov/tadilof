package knn.hnsw;

import lombok.ToString;
import lombok.extern.java.Log;

import java.util.Comparator;

@ToString(exclude = "distanceComparator")
@Log
public class NodeIdAndDistance implements Comparable<NodeIdAndDistance> {

    final int nodeId;
    final Double distance;
    final Comparator<Double> distanceComparator;

    NodeIdAndDistance(int nodeId, Double distance, Comparator<Double> distanceComparator) {
        this.nodeId = nodeId;
        this.distance = distance;
        this.distanceComparator = distanceComparator;
    }

    @Override
    public int compareTo(NodeIdAndDistance o) {
        return distanceComparator.compare(distance, o.distance);
    }
}
