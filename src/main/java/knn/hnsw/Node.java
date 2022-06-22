package knn.hnsw;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.ToString;
import org.apache.commons.math3.util.Pair;
import org.eclipse.collections.api.list.primitive.ImmutableIntList;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.slf4j.Logger;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentSkipListSet;

@Getter
@ToString
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
public class Node<TItem> implements Serializable {

    private static final long serialVersionUID = 1L;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Node.class);

    @EqualsAndHashCode.Include
    final int id;
    private final MutableIntList[] connections;
    private final ConcurrentSkipListSet<Pair<Node<TItem>, Double>> knn;
    private final ConcurrentSkipListSet<Pair<Node<TItem>, Double>> rknn;

    volatile TItem item;

    volatile boolean deleted;

    Node(int id, MutableIntList[] connections, TItem item, boolean deleted) {
        this.id = id;
        this.connections = connections;
        this.item = item;
        this.deleted = deleted;
        log.debug("NODE: {}", id);
        Comparator<Pair<Node<TItem>, Double>> comparator = new Comparator<>() {
            @Override
            public int compare(Pair<Node<TItem>, Double> o1, Pair<Node<TItem>, Double> o2) {
                int result = Comparator.<Double>reverseOrder().compare(o1.getSecond(), o2.getSecond());
                if (result == 0) {
                    return Comparator.<Integer>reverseOrder().compare(o1.getFirst().id, o2.getFirst().id);
                }
                return result;
            }
        };
        knn = new ConcurrentSkipListSet<>(comparator);
        rknn = new ConcurrentSkipListSet<>(comparator);
    }

    int maxLevel() {
        return this.connections.length - 1;
    }

    public void addConnection(int level, int neighbourId) {
        log.debug("{}:CONNECT: {} -> {}", level, id, neighbourId);
        connections[level].add(neighbourId);
    }

    public void clear(int level) {
        log.debug("{}:CLEAR: {}", level, id);
        connections[level].clear();
    }

    public ImmutableIntList getConnections(int level) {
        if (connections.length <= level) {
            return new IntArrayList().toImmutable();
        }
        return connections[level].toImmutable();
    }

    public MutableIntList[] getConnections() {
        return connections;
    }

    void writeNode(ObjectOutputStream oos) throws IOException {
        oos.writeInt(this.id);
            oos.writeInt(this.getConnections().length);

            for (MutableIntList connections : this.getConnections()) {
                oos.writeInt(connections.size());
                for (int j = 0; j < connections.size(); j++) {
                    oos.writeInt(connections.get(j));
                }
            }
//            itemSerializer.write(this.item, oos);
            oos.writeBoolean(this.deleted);

    }

    public void removeRKnn(Pair<Node<TItem>, Double> id) {
        rknn.remove(id);
    }

    public void addRKnn(Pair<Node<TItem>, Double> rk) {
        rknn.add(rk);
    }

    public synchronized SortedSet<Pair<Node<TItem>, Double>> getSortedNeighbors() {
        return Collections.unmodifiableSortedSet(knn);
    }

    public void replace(Pair<Node<TItem>, Double> furthestNeighbor, Node<TItem> neighborCandidate, Double distanceToNewNode) {
        removeKnn(furthestNeighbor);
        addKnn(neighborCandidate, distanceToNewNode);
    }

    public synchronized void addKnn(Node<TItem> neighborCandidate, Double distanceToNewNode) {
        log.trace("{} add neighbor {}", this.id, neighborCandidate.id);
        knn.add(new Pair<>(neighborCandidate, distanceToNewNode));
        neighborCandidate.addRKnn(new Pair<>(this, distanceToNewNode));
    }


    private synchronized void removeKnn(Pair<Node<TItem>, Double> node) {
        log.trace("{} remove neighbor {}", this.id, node.getFirst().id);
        knn.remove(node);
        node.getFirst().removeRKnn(new Pair<>(this, node.getValue()));

    }

    public void connectTo(Node<TItem> node2) {
        log.trace("Connect {} to {}", this.id, node2.id);

        connections[0].add(node2.id);
    }
}
