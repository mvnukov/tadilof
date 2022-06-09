package knn.hnsw;

import lombok.Getter;
import lombok.ToString;
import org.eclipse.collections.api.list.primitive.ImmutableIntList;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.slf4j.Logger;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

@Getter
@ToString
public class Node<TItem> implements Serializable {

    private static final long serialVersionUID = 1L;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Node.class);
    final int id;
    private final MutableIntList[] connections;
    final Set<Integer> reverseConnections = new HashSet<>();

    volatile TItem item;

    volatile boolean deleted;

    Node(int id, MutableIntList[] connections, TItem item, boolean deleted) {
        this.id = id;
        this.connections = connections;
        this.item = item;
        this.deleted = deleted;
        log.debug("NODE: {}", id);
    }

    int maxLevel() {
        return this.connections.length - 1;
    }

    public void addNeighbor(int level, int neighbourId) {
        log.debug("{}:CONNECT: {} -> {}", level, id, neighbourId);
        connections[level].add(neighbourId);
    }

    public void clear(int level) {
        log.debug("{}:CLEAR: {}", level, id);
        connections[level].clear();
    }

    public ImmutableIntList getConnections(int level) {
        return connections[level].toImmutable();
    }

    public MutableIntList[] getConnections() {
        return connections;
    }

    public void forEachConnection(int activeLevel, Object o) {

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

    public void removeRKnn(int id) {
        reverseConnections.remove(id);
    }

    public void addRKnn(int id) {
        reverseConnections.add(id);
    }
}
