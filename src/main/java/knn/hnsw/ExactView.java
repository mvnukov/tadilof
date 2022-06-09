package knn.hnsw;

import knn.Index;
import knn.Item;
import knn.ProgressListener;
import knn.SearchResult;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.*;

class ExactView<TId, TVector, TItem extends Item<TId, TVector>, Distance> implements Index<TId, TVector, TItem, Distance> {

    private static final long serialVersionUID = 1L;

    private final HnswIndex<TId, TVector, TItem, Distance> hnswIndex;

    public ExactView(HnswIndex<TId, TVector, TItem, Distance> hnswIndex) {
        this.hnswIndex = hnswIndex;
    }

    @Override
    public int size() {
        return hnswIndex.size();
    }

    @Override
    public Optional<TItem> get(TId tId) {
        return hnswIndex.get(tId);
    }


    @Override
    public Collection<TItem> items() {
        return hnswIndex.items();
    }

    public Optional<Set<TItem>> findReverseNeighbors(TId tId) {

        Comparator<SearchResult<TItem, Distance>> comparator = Comparator.<SearchResult<TItem, Distance>>naturalOrder().reversed();

        Set<TItem> queue = new HashSet<>();

        for (int i = 0; i < hnswIndex.nodeCount; i++) {
            Node<TItem> node = hnswIndex.nodes.get(i);
            if (node == null || node.deleted || node.item.id() == tId) {
                continue;
            }
            List<SearchResult<TItem, Distance>> nearest = findNearest(node.item.vector(), hnswIndex.getEf());

            if (nearest.stream().anyMatch(sResult -> sResult.item().id().equals(tId))) {
                queue.add(node.item);
            }
        }

        return Optional.of(queue);
    }

    @Override
    public List<SearchResult<TItem, Distance>> findNearest(TVector vector, int k) {

        Comparator<SearchResult<TItem, Distance>> comparator = Comparator.<SearchResult<TItem, Distance>>naturalOrder().reversed();

        PriorityQueue<SearchResult<TItem, Distance>> queue = new PriorityQueue<>(k, comparator);

        for (int i = 0; i < hnswIndex.nodeCount; i++) {
            Node<TItem> node = hnswIndex.nodes.get(i);
            if (node == null || node.deleted) {
                continue;
            }

            Distance distance = hnswIndex.distanceFunction.distance(node.item.vector(), vector);

            SearchResult<TItem, Distance> searchResult = new SearchResult<>(node.item, distance, hnswIndex.maxValueDistanceComparator);
            queue.add(searchResult);

            if (queue.size() > k) {
                queue.poll();
            }
        }

        List<SearchResult<TItem, Distance>> results = new ArrayList<>(queue.size());

        SearchResult<TItem, Distance> result;
        while ((result = queue.poll()) != null) { // if you iterate over a priority queue the order is not guaranteed
            results.add(0, result);
        }

        return results;
    }

    @Override
    public List<SearchResult<TItem, Distance>> findNeighbors(TId tId, int k) {
        return Index.super.findNeighbors(tId, k);
    }

    @Override
    public boolean add(TItem item) {
        return hnswIndex.add(item);
    }

    @Override
    public boolean remove(TId id, long version) {
        return hnswIndex.remove(id, version);
    }

    @Override
    public void save(OutputStream out) throws IOException {
        hnswIndex.save(out);
    }

    @Override
    public void save(File file) throws IOException {
        hnswIndex.save(file);
    }

    @Override
    public void save(Path path) throws IOException {
        hnswIndex.save(path);
    }

    @Override
    public void addAll(Collection<TItem> items) throws InterruptedException {
        hnswIndex.addAll(items);
    }

    @Override
    public void addAll(Collection<TItem> items, ProgressListener listener) throws InterruptedException {
        hnswIndex.addAll(items, listener);
    }

    @Override
    public void addAll(Collection<TItem> items, int numThreads, ProgressListener listener, int progressUpdateInterval) throws InterruptedException {
        hnswIndex.addAll(items, numThreads, listener, progressUpdateInterval);
    }
}
