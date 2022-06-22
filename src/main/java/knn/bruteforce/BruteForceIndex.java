package knn.bruteforce;

import knn.DistanceFunction;
import knn.Index;
import knn.Item;
import knn.SearchResult;
import knn.util.ClassLoaderObjectInputStream;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Implementation of {@link Index} that does pairwise comparison and as such can be used as a baseline for measuring
 * approximate nearest neighbors index precision.
 *
 * @param <TId> Type of the external identifier of an item
 * @param <TVector> Type of the vector to perform distance calculation on
 * @param <TItem> Type of items stored in the index
// * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 */
public class BruteForceIndex<TId, TVector, TItem extends Item<TId, TVector>>
        implements Index<TId, TVector, TItem, Double> {

    private static final long serialVersionUID = 1L;

    private final int dimensions;
    private final DistanceFunction<TVector, Double> distanceFunction;
    private final Comparator<Double> distanceComparator;

    private final Map<TId, TItem> items;
    private final Map<TId, Long> deletedItemVersions;

    private BruteForceIndex(Builder<TVector> builder) {
        this.dimensions = builder.dimensions;
        this.distanceFunction = builder.distanceFunction;
        this.distanceComparator = builder.distanceComparator;
        this.items = new ConcurrentHashMap<>();
        this.deletedItemVersions = new ConcurrentHashMap<>();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return items.size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Optional<TItem> get(TId id) {
        return Optional.ofNullable(items.get(id));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<TItem> items() {
        return items.values();
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
     * {@inheritDoc}
     */
    @Override
    public boolean add(TItem item) {
        if (item.dimensions() != dimensions) {
            throw new IllegalArgumentException("Item does not have dimensionality of : " + dimensions);
        }
        synchronized (items) {
            TItem existingItem = items.get(item.id());

            if (existingItem != null && item.version() < existingItem.version()) {
                return false;
            }

            if (item.version() < deletedItemVersions.getOrDefault(item.id(), 0L)) {
                return false;
            }

            items.put(item.id(), item);
            return true;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean remove(TId id, long version) {
        synchronized (items) {
            TItem item = items.get(id);

            if (item == null) {
                return false;
            }

            if (version < item.version()) {
                return false;
            }
            items.remove(id);
            deletedItemVersions.put(id, version);

            return true;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<SearchResult<TItem>> findNearest(TVector vector, int k) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null.");
        }

        Comparator<SearchResult<TItem>> comparator = Comparator
                .<SearchResult<TItem>>naturalOrder()
                .reversed();

        PriorityQueue<SearchResult<TItem>> queue = new PriorityQueue<>(k, comparator);

        for (TItem item : items.values()) {
            Double distance = distanceFunction.distance(item.vector(), vector);

            SearchResult<TItem> searchResult = new SearchResult<>(item, distance, distanceComparator);
            queue.add(searchResult);

            if (queue.size() > k) {
                queue.poll();
            }
        }

        List<SearchResult<TItem>> results = new ArrayList<>(queue.size());

        SearchResult<TItem> result;
        while((result = queue.poll()) != null) { // if you iterate over a priority queue the order is not guaranteed
            results.add(0, result);
        }

        return results;
    }

    @Override
    public Optional<Set<TItem>> findReverseNeighbors(TId tId) {
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void save(OutputStream out) throws IOException {
        try(ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    /**
     * Restores a {@link BruteForceIndex} from a File.
     *
     * @param file file to restore the index from
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem> load(File file) throws IOException {
        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link BruteForceIndex} from a File.
     *
     * @param file file to restore the index from
     * @param classLoader the classloader to use
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem> load(File file, ClassLoader classLoader) throws IOException {
        return load(new FileInputStream(file), classLoader);
    }

    /**
     * Restores a {@link BruteForceIndex} from a Path.
     *
     * @param path path to restore the index from
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem> load(Path path) throws IOException {
        return load(Files.newInputStream(path));
    }

    /**
     * Restores a {@link BruteForceIndex} from a Path.
     *
     * @param path path to restore the index from
     * @param classLoader the classloader to use
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem> load(Path path, ClassLoader classLoader) throws IOException {
        return load(Files.newInputStream(path), classLoader);
    }

    /**
     * Restores a {@link BruteForceIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem> load(InputStream inputStream) throws IOException {
        return load(inputStream, Thread.currentThread().getContextClassLoader());
    }

    /**
     * Restores a {@link BruteForceIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     * @param classLoader the classloader to use
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem> load(InputStream inputStream, ClassLoader classLoader) throws IOException {

        try(ObjectInputStream ois = new ClassLoaderObjectInputStream(classLoader, inputStream)) {
            return (BruteForceIndex<TId, TVector, TItem>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
    }

    /**
     * Start the process of building a new BruteForce index.
     *
     * @param dimensions the dimensionality of the vectors stored in the index
     * @param distanceFunction the distance function
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector, TDistance extends Comparable<Double>>
        Builder <TVector> newBuilder(int dimensions, DistanceFunction<TVector, Double> distanceFunction) {

        Comparator<Double> distanceComparator = Comparator.naturalOrder();
        return new Builder<>(dimensions, distanceFunction, distanceComparator);
    }

    /**
     * Start the process of building a new BruteForce index.
     *
     * @param dimensions the dimensionality of the vectors stored in the index
     * @param distanceFunction the distance function
     * @param distanceComparator comparator for distances
     * @param <TVector> Type of the vector to perform distance calculation on
//     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector> Builder <TVector> newBuilder(int dimensions, DistanceFunction<TVector, Double> distanceFunction, Comparator<Double> distanceComparator) {

        return new Builder<>(dimensions, distanceFunction, distanceComparator);
    }

    /**
     * Builder for initializing an {@link BruteForceIndex} instance.
     *
     * @param <TVector> Type of the vector to perform distance calculation on
//     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
    public static class Builder <TVector> {

        private final int dimensions;

        private final DistanceFunction<TVector, Double> distanceFunction;

        private final Comparator<Double> distanceComparator;

        Builder(int dimensions, DistanceFunction<TVector, Double> distanceFunction, Comparator<Double> distanceComparator) {
            this.dimensions = dimensions;
            this.distanceFunction = distanceFunction;
            this.distanceComparator = distanceComparator;
        }

        /**
         * Builds the BruteForceIndex instance.
         *
         * @param <TId> Type of the external identifier of an item
         * @param <TItem> implementation of the Item interface
         * @return the brute force index instance
         */
        public <TId, TItem extends Item<TId, TVector>> BruteForceIndex<TId, TVector, TItem> build() {
            return new BruteForceIndex<>(this);
        }

    }

}
