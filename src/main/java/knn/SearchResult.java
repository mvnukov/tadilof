package knn;

import knn.hnsw.Node;
import org.apache.commons.math3.util.Pair;

import java.io.Serializable;
import java.util.Comparator;
import java.util.Objects;

/**
 * Result of a nearest neighbour search.
 *
 * @param <TItem> type of the item returned
// * @param <Double> type of the distance returned by the configured distance function
 */
public class SearchResult<TItem>
        implements Comparable<SearchResult<TItem>>, Serializable {

    private static final long serialVersionUID = 1L;

    private final Double distance;

    private final TItem item;

    private final Comparator<Double> distanceComparator;

    /**
     * Constructs a new SearchResult instance.
     *
     * @param item the item
     * @param distance the distance from the search query
     * @param distanceComparator used to compare distances
     */
    public SearchResult(TItem item, Double distance, Comparator<Double> distanceComparator) {
        this.item = item;
        this.distance = distance;
        this.distanceComparator = distanceComparator;
    }

    /**
     * Returns the item.
     *
     * @return the item
     */
    public TItem item() {
        return item;
    }

    /**
     * Returns the distance from the search query.
     *
     * @return the distance from the search query
     */
    public Double distance() {
        return distance;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int compareTo(SearchResult<TItem> o) {
        return distanceComparator.compare(distance, o.distance);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SearchResult<?> that = (SearchResult<?>) o;
        return Objects.equals(distance, that.distance) &&
                Objects.equals(item, that.item);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        return Objects.hash(distance, item);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return "SearchResult{" +
                "distance=" + distance +
                ", item=" + item +
                '}';
    }

    /**
     * Convenience method for creating search results who's distances are Comparable.
     *
     * @param item the item
     * @param distance the distance from the search query
     * @param <TItem> type of the item returned
     * @param <TDistance> type of the distance returned by the configured distance function
     * @return new SearchResult instance
     */
    public static<TItem, TDistance extends Comparable<TDistance>> SearchResult<TItem> create(TItem item, Double distance) {
        return new SearchResult<>(item, distance, Comparator.naturalOrder());
    }
}