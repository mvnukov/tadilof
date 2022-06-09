package knn.hnsw;


import knn.*;
import knn.util.BitSet;
import knn.util.*;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.collections.api.list.primitive.ImmutableIntList;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.api.map.primitive.MutableObjectIntMap;
import org.eclipse.collections.api.map.primitive.MutableObjectLongMap;
import org.eclipse.collections.api.tuple.primitive.ObjectIntPair;
import org.eclipse.collections.api.tuple.primitive.ObjectLongPair;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectLongHashMap;
import org.slf4j.Logger;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

/**
 * Implementation of {@link Index} that implements the hnsw algorithm.
 *
 * @param <TId>      Type of the external identifier of an item
 * @param <TVector>  Type of the vector to perform distance calculation on
 * @param <Item>     Type of items stored in the index
 * @param <Distance> Type of distance between items (expect any numeric type: float, double, int, ..)
 * @see <a href="https://arxiv.org/abs/1603.09320">
 * Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</a>
 */
@Slf4j
public class HnswIndex<TId, TVector, Item extends knn.Item<TId, TVector>, Distance>
        implements Index<TId, TVector, Item, Distance> {

    private static final byte VERSION_1 = 0x01;

    private static final long serialVersionUID = 1L;

    private static final int NO_NODE_ID = -1;

    DistanceFunction<TVector, Distance> distanceFunction;
    private Comparator<Distance> distanceComparator;
    MaxValueComparator<Distance> maxValueDistanceComparator;

    private int dimensions;
    private int maxItemCount;
    private int m;
    private int maxM;
    private int maxM0;
    private double levelLambda;
    private int ef;
    private int efConstruction;
    private boolean removeEnabled;

    int nodeCount;

    private volatile Node<Item> entryPoint;

    AtomicReferenceArray<Node<Item>> nodes;
    private MutableObjectIntMap<TId> lookup;
    private MutableObjectLongMap<TId> deletedItemVersions;
    private Map<TId, Object> locks;

    private ObjectSerializer<TId> itemIdSerializer;
    private ObjectSerializer<Item> itemSerializer;

    private ReentrantLock globalLock;

    private GenericObjectPool<BitSet> visitedBitSetPool;

    private BitSet excludedCandidates;

    private ExactView<TId, TVector, Item, Distance> exactView;

    private HnswIndex(RefinedBuilder<TId, TVector, Item, Distance> builder) {

        this.dimensions = builder.dimensions;
        this.maxItemCount = builder.maxItemCount;
        this.distanceFunction = builder.distanceFunction;
        this.distanceComparator = builder.distanceComparator;
        this.maxValueDistanceComparator = new MaxValueComparator<>(this.distanceComparator);

        this.m = builder.m;
        this.maxM = builder.m;
        this.levelLambda = 1 / Math.log(this.m);
        this.efConstruction = Math.max(builder.efConstruction, m);
        this.ef = builder.ef;
        this.maxM0 = ef;

        this.removeEnabled = builder.removeEnabled;

        this.nodes = new AtomicReferenceArray<>(this.maxItemCount);

        this.lookup = new ObjectIntHashMap<>();
        this.deletedItemVersions = new ObjectLongHashMap<>();
        this.locks = new HashMap<>();

        this.itemIdSerializer = builder.itemIdSerializer;
        this.itemSerializer = builder.itemSerializer;

        this.globalLock = new ReentrantLock();

        this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount), Runtime.getRuntime().availableProcessors());

        this.excludedCandidates = new SynchronizedBitSet(new ArrayBitSet(this.maxItemCount));

        this.exactView = new ExactView(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        globalLock.lock();
        try {
            return lookup.size();
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Optional<Item> get(TId id) {
        globalLock.lock();
        try {
            int nodeId = lookup.getIfAbsent(id, NO_NODE_ID);

            if (nodeId == NO_NODE_ID) {
                return Optional.empty();
            } else {
                return Optional.of(nodes.get(nodeId).item);
            }
        } finally {
            globalLock.unlock();
        }
    }

    public Optional<Node<Item>> getNode(TId id) {
        globalLock.lock();
        try {
            int nodeId = lookup.getIfAbsent(id, NO_NODE_ID);

            if (nodeId == NO_NODE_ID) {
                return Optional.empty();
            } else {
                return Optional.of(nodes.get(nodeId));
            }
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<Item> items() {
        globalLock.lock();
        try {
            List<Item> results = new ArrayList<>(size());

            Iterator<Item> iter = new ItemIterator();

            while (iter.hasNext()) {
                results.add(iter.next());
            }

            return results;
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean remove(TId id, long version) {

        if (!removeEnabled) {
            return false;
        }

        globalLock.lock();

        try {
            int internalNodeId = lookup.getIfAbsent(id, NO_NODE_ID);

            if (internalNodeId == NO_NODE_ID) {
                return false;
            }

            Node<Item> node = nodes.get(internalNodeId);

            if (version < node.item.version()) {
                return false;
            }

            node.deleted = true;

            lookup.remove(id);

            deletedItemVersions.put(id, version);

            return true;
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean add(Item item) {
        if (item.dimensions() != dimensions) {
            throw new IllegalArgumentException("Item does not have dimensionality of : " + dimensions);
        }

        int randomLevel = assignLevel(item.id(), this.levelLambda);

        IntArrayList[] connections = new IntArrayList[randomLevel + 1];

        for (int level = 0; level <= randomLevel; level++) {
            int levelM = randomLevel == 0 ? maxM0 : maxM;
            connections[level] = new IntArrayList(levelM);
        }

        globalLock.lock();

        try {
            int existingNodeId = lookup.getIfAbsent(item.id(), NO_NODE_ID);

            if (existingNodeId != NO_NODE_ID) {

                if (!removeEnabled) {
                    return false;
                }

                Node<Item> node = nodes.get(existingNodeId);

                if (item.version() < node.item.version()) {
                    return false;
                }

                if (Objects.deepEquals(node.item.vector(), item.vector())) {
                    node.item = item;
                    return true;
                } else {
                    remove(item.id(), item.version());
                }

            } else if (item.version() < deletedItemVersions.getIfAbsent(item.id(), -1)) {
                return false;
            }

            if (nodeCount >= this.maxItemCount) {
                throw new SizeLimitExceededException("The number of elements exceeds the specified limit.");
            }

            int newNodeId = nodeCount++;

            excludedCandidates.add(newNodeId);

            Node<Item> newNode = new Node<>(newNodeId, connections, item, false);

            nodes.set(newNodeId, newNode);
            lookup.put(item.id(), newNodeId);
            deletedItemVersions.remove(item.id());

            Object lock = locks.computeIfAbsent(item.id(), k -> new Object());

            Node<Item> entryPointCopy = entryPoint;

            try {
                synchronized (lock) {
                    synchronized (newNode) {

                        if (entryPoint != null && randomLevel <= entryPoint.maxLevel()) {
                            globalLock.unlock();
                        }

                        Node<Item> currObj = entryPointCopy;

                        if (currObj != null) {

                            if (newNode.maxLevel() < entryPointCopy.maxLevel()) {

                                Distance curDist = distanceFunction.distance(item.vector(), currObj.item.vector());

                                for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > newNode.maxLevel(); activeLevel--) {

                                    boolean changed = true;

                                    while (changed) {
                                        changed = false;

                                        synchronized (currObj) {
                                            ImmutableIntList candidateConnections = currObj.getConnections(activeLevel);

                                            for (int i = 0; i < candidateConnections.size(); i++) {

                                                int candidateId = candidateConnections.get(i);

                                                Node<Item> candidateNode = nodes.get(candidateId);

                                                Distance candidateDistance = distanceFunction.distance(item.vector(), candidateNode.item.vector());

                                                if (lt(candidateDistance, curDist)) {
                                                    curDist = candidateDistance;
                                                    currObj = candidateNode;
                                                    changed = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            for (int level = Math.min(randomLevel, entryPointCopy.maxLevel()); level >= 0; level--) {
                                PriorityQueue<NodeIdAndDistance<Distance>> topCandidates = searchBaseLayer(currObj, item.vector(), efConstruction, level);

                                if (entryPointCopy.deleted) {
                                    Distance distance = distanceFunction.distance(item.vector(), entryPointCopy.item.vector());
                                    topCandidates.add(new NodeIdAndDistance<>(entryPointCopy.id, distance, maxValueDistanceComparator));

                                    if (topCandidates.size() > efConstruction) {
                                        topCandidates.poll();
                                    }
                                }


                                mutuallyConnectNewElement(newNode, topCandidates, level);

                            }

                            notifyNeighbors(newNode);
                        }

                        // zoom out to the highest level
                        if (entryPoint == null || newNode.maxLevel() > entryPointCopy.maxLevel()) {
                            // this is thread safe because we get the global lock when we add a level
                            this.entryPoint = newNode;
                        }

                        return true;
                    }
                }
            } finally {
                excludedCandidates.remove(newNodeId);
            }
        } finally {
            if (globalLock.isHeldByCurrentThread()) {
                globalLock.unlock();
            }
        }
    }

    private void notifyNeighborsClear(Node<Item> newNode) {
        var neighbors = newNode.getConnections(0);
        neighbors.forEach(neighborNodeId -> {
            var neighborNode = nodes.get(neighborNodeId);

            neighborNode.removeRKnn(newNode.id);
        });

    }

    private void notifyNeighbors(Node<Item> newNode) {
        var neighbors = newNode.getConnections(0);
        neighbors.forEach(neighborNodeId -> {
            var neighborNode = nodes.get(neighborNodeId);

            neighborNode.addRKnn(newNode.id);
        });
    }

    private void mutuallyConnectNewElement(Node<Item> newNode, PriorityQueue<NodeIdAndDistance<Distance>> topCandidates, int level) {

        int bestN = level == 0 ? this.maxM0 : this.maxM;

        int newNodeId = newNode.id;
        TVector newItemVector = newNode.item.vector();

        getNeighborsByHeuristic3(topCandidates, m);

        log.trace("topCandidates: {}", topCandidates);
        PriorityQueue<NodeIdAndDistance<Distance>> topCandidates2 =
                new PriorityQueue<>(Comparator.<NodeIdAndDistance<Distance>>naturalOrder());

        topCandidates2.addAll(topCandidates);

        while (!topCandidates2.isEmpty()) {
            int selectedNeighbourId = topCandidates2.poll().nodeId;

            if (excludedCandidates.contains(selectedNeighbourId)) {
                continue;
            }

            newNode.addNeighbor(level, selectedNeighbourId);

            Node<Item> neighbourNode = nodes.get(selectedNeighbourId);

            synchronized (neighbourNode) {

                TVector neighbourVector = neighbourNode.item.vector();

                ImmutableIntList neighbourConnectionsAtLevel = neighbourNode.getConnections(level);

                if (neighbourConnectionsAtLevel.size() < bestN) {
                    neighbourNode.addNeighbor(level, newNodeId);
                } else {
                    // finding the "weakest" element to replace it with the new one

                    Distance dMax = distanceFunction.distance(newItemVector, neighbourNode.item.vector());

                    Comparator<NodeIdAndDistance<Distance>> comparator = Comparator.<NodeIdAndDistance<Distance>>naturalOrder().reversed();

                    PriorityQueue<NodeIdAndDistance<Distance>> candidates = new PriorityQueue<>(comparator);
                    candidates.add(new NodeIdAndDistance<>(newNodeId, dMax, maxValueDistanceComparator));

                    neighbourConnectionsAtLevel.forEach(id -> {
                        Distance dist = distanceFunction.distance(neighbourVector, nodes.get(id).item.vector());

                        candidates.add(new NodeIdAndDistance<>(id, dist, maxValueDistanceComparator));
                    });

                    getNeighborsByHeuristic3(candidates, bestN);

                    if (level == 0) {
                        notifyNeighborsClear(neighbourNode);
                    }

                    neighbourNode.clear(level);

                    while (!candidates.isEmpty()) {
                        neighbourNode.addNeighbor(level, candidates.poll().nodeId);
                    }
                }
                if (level == 0) {
                    notifyNeighbors(neighbourNode);
                }
            }
        }
    }
    private void getNeighborsByHeuristic3(PriorityQueue<NodeIdAndDistance<Distance>> topCandidates, int m) {
        while(topCandidates.size() > m) {
            topCandidates.poll();
        }
    }

    private void getNeighborsByHeuristic2(PriorityQueue<NodeIdAndDistance<Distance>> topCandidates, int m) {

        if (topCandidates.size() < m) {
            return;
        }

        PriorityQueue<NodeIdAndDistance<Distance>> queueClosest = new PriorityQueue<>();
        List<NodeIdAndDistance<Distance>> returnList = new ArrayList<>();

        while (!topCandidates.isEmpty()) {
            queueClosest.add(topCandidates.poll());
        }

        while (!queueClosest.isEmpty()) {
            if (returnList.size() >= m) {
                break;
            }

            NodeIdAndDistance<Distance> currentPair = queueClosest.poll();

            Distance distToQuery = currentPair.distance;

            boolean good = true;
            for (NodeIdAndDistance<Distance> secondPair : returnList) {

                Distance curdist = distanceFunction.distance(nodes.get(secondPair.nodeId).item.vector(), nodes.get(currentPair.nodeId).item.vector());

                if (lt(curdist, distToQuery)) {
                    good = false;
                    break;
                }

            }
            if (good) {
                returnList.add(currentPair);
            }
        }

        topCandidates.addAll(returnList);
    }


    public Optional<Set<Item>> findReverseNeighbors(TId tId) {
        Optional<Set<Item>> o = getNode(tId)
                .map(node -> node.reverseConnections)
                .map(reverseConnections -> reverseConnections.stream()
                        .map(reverseNeighbor -> {
                            Node<Item> tItemNode = nodes.get(reverseNeighbor);
                            return tItemNode.item;
                        })
                        .collect(Collectors.toSet()));
        return o;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<SearchResult<Item, Distance>> findNearest(TVector destination, int k) {

        if (entryPoint == null) {
            return Collections.emptyList();
        }

        Node<Item> entryPointCopy = entryPoint;
        Node<Item> currObj = entryPointCopy;
        Distance curDist = distanceFunction.distance(destination, currObj.item.vector());

        for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > 0; activeLevel--) {

            boolean changed = true;

            while (changed) {
                changed = false;

                synchronized (currObj) {
                    ImmutableIntList candidateConnections = currObj.getConnections(activeLevel);

                    for (int i = 0; i < candidateConnections.size(); i++) {
                        int candidateId = candidateConnections.get(i);

                        Distance candidateDistance = distanceFunction.distance(destination, nodes.get(candidateId).item.vector());
                        if (lt(candidateDistance, curDist)) {
                            curDist = candidateDistance;
                            currObj = nodes.get(candidateId);
                            changed = true;
                        }
                    }
                }

            }
        }

        PriorityQueue<NodeIdAndDistance<Distance>> topCandidates = searchBaseLayer(currObj, destination, Math.max(ef, k), 0);

        while (topCandidates.size() > k) {
            topCandidates.poll();
        }

        List<SearchResult<Item, Distance>> results = new ArrayList<>(topCandidates.size());
        while (!topCandidates.isEmpty()) {
            NodeIdAndDistance<Distance> pair = topCandidates.poll();
            results.add(0, new SearchResult<>(nodes.get(pair.nodeId).item, pair.distance, maxValueDistanceComparator));
        }

        return results;
    }

    private PriorityQueue<NodeIdAndDistance<Distance>> searchBaseLayer(Node<Item> entryPointNode, TVector destination, int k, int layer) {

        BitSet visitedBitSet = visitedBitSetPool.borrowObject();

        try {
            PriorityQueue<NodeIdAndDistance<Distance>> topCandidates = new PriorityQueue<>(Comparator.<NodeIdAndDistance<Distance>>naturalOrder().reversed());
            PriorityQueue<NodeIdAndDistance<Distance>> candidateSet = new PriorityQueue<>();

            Distance lowerBound;

            if (!entryPointNode.deleted) {
                Distance distance = distanceFunction.distance(destination, entryPointNode.item.vector());
                NodeIdAndDistance<Distance> pair = new NodeIdAndDistance<>(entryPointNode.id, distance, maxValueDistanceComparator);

                topCandidates.add(pair);
                lowerBound = distance;
                candidateSet.add(pair);

            } else {
                lowerBound = MaxValueComparator.maxValue();
                NodeIdAndDistance<Distance> pair = new NodeIdAndDistance<>(entryPointNode.id, lowerBound, maxValueDistanceComparator);
                candidateSet.add(pair);
            }

            visitedBitSet.add(entryPointNode.id);

            while (!candidateSet.isEmpty()) {

                NodeIdAndDistance<Distance> currentPair = candidateSet.poll();

                if (gt(currentPair.distance, lowerBound)) {
                    break;
                }

                Node<Item> node = nodes.get(currentPair.nodeId);

                synchronized (node) {

                    ImmutableIntList candidates = node.getConnections(layer);

                    for (int i = 0; i < candidates.size(); i++) {

                        int candidateId = candidates.get(i);

                        if (!visitedBitSet.contains(candidateId)) {

                            visitedBitSet.add(candidateId);

                            Node<Item> candidateNode = nodes.get(candidateId);

                            Distance candidateDistance = distanceFunction.distance(destination, candidateNode.item.vector());

                            if (topCandidates.size() < k || gt(lowerBound, candidateDistance)) {

                                NodeIdAndDistance<Distance> candidatePair = new NodeIdAndDistance<>(candidateId, candidateDistance, maxValueDistanceComparator);

                                candidateSet.add(candidatePair);

                                if (!candidateNode.deleted) {
                                    topCandidates.add(candidatePair);
                                }

                                if (topCandidates.size() > k) {
                                    topCandidates.poll();
                                }

                                if (!topCandidates.isEmpty()) {
                                    lowerBound = topCandidates.peek().distance;
                                }
                            }
                        }
                    }

                }
            }

            return topCandidates;
        } finally {
            visitedBitSet.clear();
            visitedBitSetPool.returnObject(visitedBitSet);
        }
    }

    /**
     * Creates a read only view on top of this index that uses pairwise comparision when doing distance search. And as
     * such can be used as a baseline for assessing the precision of the index.
     * Searches will be really slow but give the correct result every time.
     *
     * @return read only view on top of this index that uses pairwise comparision when doing distance search
     */
    public Index<TId, TVector, Item, Distance> asExactIndex() {
        return exactView;
    }

    /**
     * Returns the dimensionality of the items stored in this index.
     *
     * @return the dimensionality of the items stored in this index
     */
    public int getDimensions() {
        return dimensions;
    }

    /**
     * Returns the number of bi-directional links created for every new element during construction.
     *
     * @return the number of bi-directional links created for every new element during construction
     */
    public int getM() {
        return m;
    }

    /**
     * The size of the dynamic list for the nearest neighbors (used during the search)
     *
     * @return The size of the dynamic list for the nearest neighbors
     */
    public int getEf() {
        return ef;
    }

    /**
     * Set the size of the dynamic list for the nearest neighbors (used during the search)
     *
     * @param ef The size of the dynamic list for the nearest neighbors
     */
    public void setEf(int ef) {
        this.ef = ef;
    }

    /**
     * Returns the parameter has the same meaning as ef, but controls the index time / index precision.
     *
     * @return the parameter has the same meaning as ef, but controls the index time / index precision
     */
    public int getEfConstruction() {
        return efConstruction;
    }

    /**
     * Returns the distance function.
     *
     * @return the distance function
     */
    public DistanceFunction<TVector, Distance> geDistanceFunction() {
        return distanceFunction;
    }


    /**
     * Returns the comparator used to compare distances.
     *
     * @return the comparator used to compare distance
     */
    public Comparator<Distance> geDistanceComparator() {
        return distanceComparator;
    }

    /**
     * Returns if removes are enabled.
     *
     * @return true if removes are enabled for this index.
     */
    public boolean isRemoveEnabled() {
        return removeEnabled;
    }

    /**
     * Returns the maximum number of items the index can hold.
     *
     * @return the maximum number of items the index can hold
     */
    public int getMaxItemCount() {
        return maxItemCount;
    }

    /**
     * Returns the serializer used to serialize item id's when saving the index.
     *
     * @return the serializer used to serialize item id's when saving the index
     */
    public ObjectSerializer<TId> getItemIdSerializer() {
        return itemIdSerializer;
    }

    /**
     * Returns the serializer used to serialize items when saving the index.
     *
     * @return the serializer used to serialize items when saving the index
     */
    public ObjectSerializer<Item> getItemSerializer() {
        return itemSerializer;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void save(OutputStream out) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.writeByte(VERSION_1);
        oos.writeInt(dimensions);
        oos.writeObject(distanceFunction);
        oos.writeObject(distanceComparator);
        oos.writeObject(itemIdSerializer);
        oos.writeObject(itemSerializer);
        oos.writeInt(maxItemCount);
        oos.writeInt(m);
        oos.writeInt(maxM);
        oos.writeInt(maxM0);
        oos.writeDouble(levelLambda);
        oos.writeInt(ef);
        oos.writeInt(efConstruction);
        oos.writeBoolean(removeEnabled);
        oos.writeInt(nodeCount);
        writeMutableObjectIntMap(oos, lookup);
        writeMutableObjectLongMap(oos, deletedItemVersions);
        writeNodesArray(oos, nodes);
        oos.writeInt(entryPoint == null ? -1 : entryPoint.id);
    }

    @SuppressWarnings("unchecked")
    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        @SuppressWarnings("unused") byte version = ois.readByte(); // for coping with future incompatible serialization
        this.dimensions = ois.readInt();
        this.distanceFunction = (DistanceFunction<TVector, Distance>) ois.readObject();
        this.distanceComparator = (Comparator<Distance>) ois.readObject();
        this.maxValueDistanceComparator = new MaxValueComparator<>(distanceComparator);
        this.itemIdSerializer = (ObjectSerializer<TId>) ois.readObject();
        this.itemSerializer = (ObjectSerializer<Item>) ois.readObject();

        this.maxItemCount = ois.readInt();
        this.m = ois.readInt();
        this.maxM = ois.readInt();
        this.maxM0 = ois.readInt();
        this.levelLambda = ois.readDouble();
        this.ef = ois.readInt();
        this.efConstruction = ois.readInt();
        this.removeEnabled = ois.readBoolean();
        this.nodeCount = ois.readInt();
        this.lookup = readMutableObjectIntMap(ois, itemIdSerializer);
        this.deletedItemVersions = readMutableObjectLongMap(ois, itemIdSerializer);
        this.nodes = readNodesArray(ois, itemSerializer, maxM0, maxM);

        int entrypointNodeId = ois.readInt();
        this.entryPoint = entrypointNodeId == -1 ? null : nodes.get(entrypointNodeId);

        this.globalLock = new ReentrantLock();
        this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount), Runtime.getRuntime().availableProcessors());
        this.excludedCandidates = new SynchronizedBitSet(new ArrayBitSet(this.maxItemCount));
        this.locks = new HashMap<>();
        this.exactView = new ExactView(this);
    }

    private void writeMutableObjectIntMap(ObjectOutputStream oos, MutableObjectIntMap<TId> map) throws IOException {
        oos.writeInt(map.size());

        for (ObjectIntPair<TId> pair : map.keyValuesView()) {
            itemIdSerializer.write(pair.getOne(), oos);
            oos.writeInt(pair.getTwo());
        }
    }

    private void writeMutableObjectLongMap(ObjectOutputStream oos, MutableObjectLongMap<TId> map) throws IOException {
        oos.writeInt(map.size());

        for (ObjectLongPair<TId> pair : map.keyValuesView()) {
            itemIdSerializer.write(pair.getOne(), oos);
            oos.writeLong(pair.getTwo());
        }
    }

    private void writeNodesArray(ObjectOutputStream oos, AtomicReferenceArray<Node<Item>> nodes) throws IOException {
        oos.writeInt(nodes.length());
        for (int i = 0; i < nodes.length(); i++) {
            if (nodes.get(i) == null) {
                oos.writeInt(-1);
            } else {
                nodes.get(i).writeNode(oos);
            }
        }
    }

        /**
         * Restores a {@link HnswIndex} from a File.
         *
         * @param file       File to restore the index from
         * @param <TId>      Type of the external identifier of an item
         * @param <TVector>  Type of the vector to perform distance calculation on
         * @param <TItem>    Type of items stored in the index
         * @param <Distance> Type of distance between items (expect any numeric type: float, double, int, ..)
         * @return The restored index
         * @throws IOException in case of an I/O exception
         */
        public static <TId, TVector, TItem extends knn.Item<TId, TVector>,
        Distance > HnswIndex < TId, TVector, TItem, Distance > load(File file) throws IOException {
            return load(new FileInputStream(file));
        }

        /**
         * Restores a {@link HnswIndex} from a File.
         *
         * @param file        File to restore the index from
         * @param classLoader the classloader to use
         * @param <TId>       Type of the external identifier of an item
         * @param <TVector>   Type of the vector to perform distance calculation on
         * @param <TItem>     Type of items stored in the index
         * @param <Distance>  Type of distance between items (expect any numeric type: float, double, int, ..)
         * @return The restored index
         * @throws IOException in case of an I/O exception
         */
        public static <TId, TVector, TItem extends knn.Item<TId, TVector>,
        Distance > HnswIndex < TId, TVector, TItem, Distance >
                load(File file, ClassLoader classLoader) throws IOException {
            return load(new FileInputStream(file), classLoader);
        }

        /**
         * Restores a {@link HnswIndex} from a Path.
         *
         * @param path       Path to restore the index from
         * @param <TId>      Type of the external identifier of an item
         * @param <TVector>  Type of the vector to perform distance calculation on
         * @param <TItem>    Type of items stored in the index
         * @param <Distance> Type of distance between items (expect any numeric type: float, double, int, ..)
         * @return The restored index
         * @throws IOException in case of an I/O exception
         */
        public static <TId, TVector, TItem extends knn.Item<TId, TVector>,
        Distance > HnswIndex < TId, TVector, TItem, Distance >
                load(Path path) throws IOException {
            return load(Files.newInputStream(path));
        }

        /**
         * Restores a {@link HnswIndex} from a Path.
         *
         * @param path        Path to restore the index from
         * @param classLoader the classloader to use
         * @param <TId>       Type of the external identifier of an item
         * @param <TVector>   Type of the vector to perform distance calculation on
         * @param <TItem>     Type of items stored in the index
         * @param <Distance>  Type of distance between items (expect any numeric type: float, double, int, ..)
         * @return The restored index
         * @throws IOException in case of an I/O exception
         */
        public static <TId, TVector, TItem extends knn.Item<TId, TVector>,
        Distance > HnswIndex < TId, TVector, TItem, Distance >
                load(Path path, ClassLoader classLoader) throws IOException {
            return load(Files.newInputStream(path), classLoader);
        }

        /**
         * Restores a {@link HnswIndex} from an InputStream.
         *
         * @param inputStream InputStream to restore the index from
         * @param <TId>       Type of the external identifier of an item
         * @param <TVector>   Type of the vector to perform distance calculation on
         * @param <TItem>     Type of items stored in the index
         * @param <Distance>  Type of distance between items (expect any numeric type: float, double, int, ...).
         * @return The restored index
         * @throws IOException              in case of an I/O exception
         * @throws IllegalArgumentException in case the file cannot be read
         */
        public static <TId, TVector, TItem extends knn.Item<TId, TVector>,
        Distance > HnswIndex < TId, TVector, TItem, Distance >
                load(InputStream inputStream) throws IOException {
            return load(inputStream, Thread.currentThread().getContextClassLoader());
        }

        /**
         * Restores a {@link HnswIndex} from an InputStream.
         *
         * @param inputStream InputStream to restore the index from
         * @param classLoader the classloader to use
         * @param <TId>       Type of the external identifier of an item
         * @param <TVector>   Type of the vector to perform distance calculation on
         * @param <TItem>     Type of items stored in the index
         * @param <Distance>  Type of distance between items (expect any numeric type: float, double, int, ...).
         * @return The restored index
         * @throws IOException              in case of an I/O exception
         * @throws IllegalArgumentException in case the file cannot be read
         */
        @SuppressWarnings("unchecked")
        public static <TId, TVector, TItem extends knn.Item<TId, TVector>,
        Distance > HnswIndex < TId, TVector, TItem, Distance >
                load(InputStream inputStream, ClassLoader classLoader) throws IOException {

            try (ObjectInputStream ois = new ClassLoaderObjectInputStream(classLoader, inputStream)) {
                return (HnswIndex<TId, TVector, TItem, Distance>) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new IllegalArgumentException("Could not read input file.", e);
            }
        }

        private static IntArrayList readIntArrayList (ObjectInputStream ois,int initialSize) throws IOException {
            int size = ois.readInt();

            IntArrayList list = new IntArrayList(initialSize);

            for (int j = 0; j < size; j++) {
                list.add(ois.readInt());
            }

            return list;
        }

        private static <
        TItem > Node < TItem > readNode(ObjectInputStream ois, ObjectSerializer < TItem > itemSerializer, int maxM0,
        int maxM) throws IOException, ClassNotFoundException {

            int id = ois.readInt();

            if (id == -1) {
                return null;
            } else {
                int connectionsSize = ois.readInt();

                MutableIntList[] connections = new MutableIntList[connectionsSize];

                for (int i = 0; i < connectionsSize; i++) {
                    int levelM = i == 0 ? maxM0 : maxM;
                    connections[i] = readIntArrayList(ois, levelM);
                }

                TItem item = itemSerializer.read(ois);

                boolean deleted = ois.readBoolean();

                return new Node<>(id, connections, item, deleted);
            }
        }

        private static <
        TItem > AtomicReferenceArray < Node < TItem >> readNodesArray(ObjectInputStream ois, ObjectSerializer < TItem > itemSerializer,
        int maxM0, int maxM) throws IOException, ClassNotFoundException {

            int size = ois.readInt();
            AtomicReferenceArray<Node<TItem>> nodes = new AtomicReferenceArray<>(size);

            for (int i = 0; i < nodes.length(); i++) {
                nodes.set(i, readNode(ois, itemSerializer, maxM0, maxM));
            }

            return nodes;
        }

        private static <
        TId > MutableObjectIntMap < TId > readMutableObjectIntMap(ObjectInputStream ois, ObjectSerializer < TId > itemIdSerializer) throws
        IOException, ClassNotFoundException {

            int size = ois.readInt();

            MutableObjectIntMap<TId> map = new ObjectIntHashMap<>(size);

            for (int i = 0; i < size; i++) {
                TId key = itemIdSerializer.read(ois);
                int value = ois.readInt();

                map.put(key, value);
            }
            return map;
        }

        private static <
        TId > MutableObjectLongMap < TId > readMutableObjectLongMap(ObjectInputStream ois, ObjectSerializer < TId > itemIdSerializer) throws
        IOException, ClassNotFoundException {

            int size = ois.readInt();

            MutableObjectLongMap<TId> map = new ObjectLongHashMap<>(size);

            for (int i = 0; i < size; i++) {
                TId key = itemIdSerializer.read(ois);
                long value = ois.readLong();

                map.put(key, value);
            }
            return map;
        }

        /**
         * Start the process of building a new HNSW index.
         *
         * @param dimensions       the dimensionality of the vectors stored in the index
         * @param distanceFunction the distance function
         * @param maxItemCount     maximum number of items the index can hold
         * @param <TVector>        Type of the vector to perform distance calculation on
         * @param <Distance>       Type of distance between items (expect any numeric type: float, double, int, ..)
         * @return a builder
         */
        public static <TVector, Distance extends Comparable<Distance>>Builder<TVector, Distance> newBuilder (
        int dimensions, DistanceFunction<TVector, Distance > distanceFunction,int maxItemCount){

            Comparator<Distance> distanceComparator = Comparator.naturalOrder();

            return new Builder<>(dimensions, distanceFunction, distanceComparator, maxItemCount);
        }

        /**
         * Start the process of building a new HNSW index.
         *
         * @param dimensions         the dimensionality of the vectors stored in the index
         * @param distanceFunction   the distance function
         * @param distanceComparator used to compare distances
         * @param maxItemCount       maximum number of items the index can hold
         * @param <TVector>          Type of the vector to perform distance calculation on
         * @param <Distance>         Type of distance between items (expect any numeric type: float, double, int, ..)
         * @return a builder
         */
        public static <TVector, Distance > Builder < TVector, Distance > newBuilder( int dimensions, DistanceFunction<
        TVector, Distance > distanceFunction, Comparator < Distance > distanceComparator,int maxItemCount){

            return new Builder<>(dimensions, distanceFunction, distanceComparator, maxItemCount);
        }

        private int assignLevel (TId value,double lambda){

            // by relying on the external id to come up with the level, the graph construction should be a lot mor stable
            // see : https://github.com/nmslib/hnswlib/issues/28

            int hashCode = value.hashCode();

            byte[] bytes = new byte[]{(byte) (hashCode >> 24), (byte) (hashCode >> 16), (byte) (hashCode >> 8), (byte) hashCode};

            double random = Math.abs((double) Murmur3.hash32(bytes) / (double) Integer.MAX_VALUE);

            double r = -Math.log(random) * lambda;
            return (int) r;
        }

        private boolean lt (Distance x, Distance y){
            return maxValueDistanceComparator.compare(x, y) < 0;
        }

        private boolean gt (Distance x, Distance y){
            return maxValueDistanceComparator.compare(x, y) > 0;
        }

        class ItemIterator implements Iterator<Item> {

            private int done = 0;
            private int index = 0;

            @Override
            public boolean hasNext() {
                return done < HnswIndex.this.size();
            }

            @Override
            public Item next() {
                Node<Item> node;

                do {
                    node = HnswIndex.this.nodes.get(index++);
                } while (node == null || node.deleted);

                done++;

                return node.item;
            }
        }


    static class MaxValueComparator<Distance> implements Comparator<Distance>, Serializable {

            private static final long serialVersionUID = 1L;

            private final Comparator<Distance> delegate;

            MaxValueComparator(Comparator<Distance> delegate) {
                this.delegate = delegate;
            }

            @Override
            public int compare(Distance o1, Distance o2) {
                return o1 == null ? o2 == null ? 0 : 1 : o2 == null ? -1 : delegate.compare(o1, o2);
            }

            static <Distance> Distance maxValue() {
                return null;
            }
        }

        /**
         * Base class for HNSW index builders.
         *
         * @param <TBuilder> Concrete class that extends from this builder
         * @param <TVector>  Type of the vector to perform distance calculation on
         * @param <Distance> Type of items stored in the index
         */
        public static abstract class BuilderBase<TBuilder extends BuilderBase<TBuilder, TVector, Distance>, TVector, Distance> {

            public static final int DEFAULT_M = 10;
            public static final int DEFAULT_EF = 10;
            public static final int DEFAULT_EF_CONSTRUCTION = 200;
            public static final boolean DEFAULT_REMOVE_ENABLED = false;

            int dimensions;
            DistanceFunction<TVector, Distance> distanceFunction;
            Comparator<Distance> distanceComparator;

            int maxItemCount;

            int m = DEFAULT_M;
            int ef = DEFAULT_EF;
            int efConstruction = DEFAULT_EF_CONSTRUCTION;
            boolean removeEnabled = DEFAULT_REMOVE_ENABLED;

            BuilderBase(int dimensions, DistanceFunction<TVector, Distance> distanceFunction, Comparator<Distance> distanceComparator, int maxItemCount) {

                this.dimensions = dimensions;
                this.distanceFunction = distanceFunction;
                this.distanceComparator = distanceComparator;
                this.maxItemCount = maxItemCount;
            }

            abstract TBuilder self();

            /**
             * Sets the number of bi-directional links created for every new element during construction. Reasonable range
             * for m is 2-100. Higher m work better on datasets with high intrinsic dimensionality and/or high recall,
             * while low m work better for datasets with low intrinsic dimensionality and/or low recalls. The parameter
             * also determines the algorithm's memory consumption.
             * As an example for d = 4 random vectors optimal m for search is somewhere around 6, while for high dimensional
             * datasets (word embeddings, good face descriptors), higher M are required (e.g. m = 48, 64) for optimal
             * performance at high recall. The range mM = 12-48 is ok for the most of the use cases. When m is changed one
             * has to update the other parameters. Nonetheless, ef and efConstruction parameters can be roughly estimated by
             * assuming that m  efConstruction is a constant.
             *
             * @param m the number of bi-directional links created for every new element during construction
             * @return the builder.
             */
            public TBuilder withM(int m) {
                this.m = m;
                return self();
            }

            /**
             * `
             * The parameter has the same meaning as ef, but controls the index time / index precision. Bigger efConstruction
             * leads to longer construction, but better index quality. At some point, increasing efConstruction does not
             * improve the quality of the index. One way to check if the selection of ef_construction was ok is to measure
             * a recall for M nearest neighbor search when ef = efConstruction: if the recall is lower than 0.9, then
             * there is room for improvement.
             *
             * @param efConstruction controls the index time / index precision
             * @return the builder
             */
            public TBuilder withEfConstruction(int efConstruction) {
                this.efConstruction = efConstruction;
                return self();
            }

            /**
             * The size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more
             * accurate but slower search. The value ef of can be anything between k and the size of the dataset.
             *
             * @param ef size of the dynamic list for the nearest neighbors
             * @return the builder
             */
            public TBuilder withEf(int ef) {
                this.ef = ef;
                return self();
            }

            /**
             * Call to enable support for the experimental remove operation. Indices that support removes will consume more
             * memory.
             *
             * @return the builder
             */
            public TBuilder withRemoveEnabled() {
                this.removeEnabled = true;
                return self();
            }
        }


        /**
         * Builder for initializing an {@link HnswIndex} instance.
         *
         * @param <TVector>  Type of the vector to perform distance calculation on
         * @param <Distance> Type of distance between items (expect any numeric type: float, double, int, ..)
         */
        public static class Builder<TVector, Distance> extends BuilderBase<Builder<TVector, Distance>, TVector, Distance> {

            /**
             * Constructs a new {@link Builder} instance.
             *
             * @param dimensions       the dimensionality of the vectors stored in the index
             * @param distanceFunction the distance function
             * @param maxItemCount     the maximum number of elements in the index
             */
            Builder(int dimensions, DistanceFunction<TVector, Distance> distanceFunction, Comparator<Distance> distanceComparator, int maxItemCount) {

                super(dimensions, distanceFunction, distanceComparator, maxItemCount);
            }

            @Override
            Builder<TVector, Distance> self() {
                return this;
            }

            /**
             * Register the serializers used when saving the index.
             *
             * @param itemIdSerializer serializes the key of the item
             * @param itemSerializer   serializes the
             * @param <TId>            Type of the external identifier of an item
             * @param <TItem>          implementation of the Item interface
             * @return the builder
             */
            public <TId, TItem extends knn.Item<TId, TVector>> RefinedBuilder<TId, TVector, TItem, Distance>
            withCustomSerializers(ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {
                return new RefinedBuilder<>(dimensions, distanceFunction, distanceComparator, maxItemCount, m, ef,
                        efConstruction, removeEnabled, itemIdSerializer, itemSerializer);
            }

            /**
             * Build the index that uses java object serializers to store the items when reading and writing the index.
             *
             * @param <TId>   Type of the external identifier of an item
             * @param <TItem> implementation of the Item interface
             * @return the hnsw index instance
             */
            public <TId, TItem extends knn.Item<TId, TVector>> HnswIndex<TId, TVector, TItem, Distance> build() {
                ObjectSerializer<TId> itemIdSerializer = new JavaObjectSerializer<>();
                ObjectSerializer<TItem> itemSerializer = new JavaObjectSerializer<>();

                return withCustomSerializers(itemIdSerializer, itemSerializer).build();
            }

        }

        /**
         * Extension of {@link Builder} that has knows what type of item is going to be stored in the index.
         *
         * @param <TId>      Type of the external identifier of an item
         * @param <TVector>  Type of the vector to perform distance calculation on
         * @param <TItem>    Type of items stored in the index
         * @param <Distance> Type of distance between items (expect any numeric type: float, double, int, ..)
         */
        public static class RefinedBuilder<TId, TVector, TItem extends knn.Item<TId, TVector>, Distance> extends
                BuilderBase<RefinedBuilder<TId, TVector, TItem, Distance>, TVector, Distance> {

            private ObjectSerializer<TId> itemIdSerializer;
            private ObjectSerializer<TItem> itemSerializer;

            RefinedBuilder(int dimensions, DistanceFunction<TVector, Distance> distanceFunction, Comparator<Distance> distanceComparator, int maxItemCount, int m, int ef, int efConstruction, boolean removeEnabled, ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {

                super(dimensions, distanceFunction, distanceComparator, maxItemCount);

                this.m = m;
                this.ef = ef;
                this.efConstruction = efConstruction;
                this.removeEnabled = removeEnabled;

                this.itemIdSerializer = itemIdSerializer;
                this.itemSerializer = itemSerializer;
            }

            @Override
            RefinedBuilder<TId, TVector, TItem, Distance> self() {
                return this;
            }

            /**
             * Register the serializers used when saving the index.
             *
             * @param itemIdSerializer serializes the key of the item
             * @param itemSerializer   serializes the
             * @return the builder
             */
            public RefinedBuilder<TId, TVector, TItem, Distance> withCustomSerializers(ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {

                this.itemIdSerializer = itemIdSerializer;
                this.itemSerializer = itemSerializer;

                return this;
            }

            /**
             * Build the index.
             *
             * @return the hnsw index instance
             */
            public HnswIndex<TId, TVector, TItem, Distance> build() {
                return new HnswIndex<>(this);
            }

        }

    }
