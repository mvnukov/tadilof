package knn.hnsw;

import knn.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;

class HnswIndexTest {
    private HnswIndex<String, double[], TestItem> index;

    private int maxItemCount = 300;
    private int m = 12;
    private int efConstruction = 250;
    private int ef = 10;
    private int dimensions = 2;
    private DistanceFunction<double[], Double> distanceFunction = DistanceFunctions.DOUBLE_MANHATTAN_DISTANCE;
    private ObjectSerializer<String> itemIdSerializer = new JavaObjectSerializer<>();
    private ObjectSerializer<TestItem> itemSerializer = new JavaObjectSerializer<>();

    private TestItem item1 = new TestItem("1", new double[]{0.0110f, 0.2341f}, 10);
    private TestItem item2 = new TestItem("2", new double[]{0.2300f, 0.3891f}, 10);
    private TestItem item3 = new TestItem("3", new double[]{0.4300f, 0.9891f}, 10);

    @BeforeEach
    void setUp() {
        HnswIndex<String, double[], TestItem> build = (HnswIndex<String, double[], TestItem>) HnswIndex
                .newBuilder(dimensions, distanceFunction, maxItemCount)
                .withCustomSerializers(itemIdSerializer, itemSerializer)
                .withM(m)
                .withEfConstruction(efConstruction)
                .withEf(ef)
                .withRemoveEnabled()
                .build();
        index = build;
    }

    @Test
    void returnDimensions() {
        assertThat(index.getDimensions(), is(dimensions));
    }

    @Test
    void returnM() {
        assertThat(index.getM(), is(m));
    }

    @Test
    void returnEf() {
        assertThat(index.getEf(), is(ef));
    }

    @Test
    void changeEf() {
        int newEfValue = 999;
        index.setEf(newEfValue);
        assertThat(index.getEf(), is(newEfValue));
    }

    @Test
    void returnEfConstruction() {
        assertThat(index.getEfConstruction(), is(efConstruction));
    }

    @Test
    void returnMaxItemCount() {
        assertThat(index.getMaxItemCount(), is(maxItemCount));
    }

    @Test
    void returnDistanceFunction() {
        assertThat(index.geDistanceFunction(), is(sameInstance(distanceFunction)));
    }

    @Test
    void returnsItemIdSerializer() {
        assertThat(index.getItemIdSerializer(), is(sameInstance(itemIdSerializer)));
    }

    @Test
    void returnsItemSerializer() {
        assertThat(index.getItemSerializer(), is(sameInstance(itemSerializer)));
    }

    @Test
    void returnsSize() {
        assertThat(index.size(), is(0));
        index.add2(item1);
        assertThat(index.size(), is(1));
    }

    @Test
    void addAndGet() {
        assertThat(index.get(item1.id()), is(Optional.empty()));
        index.add2(item1);
        assertThat(index.get(item1.id()), is(Optional.of(item1)));
    }

    @Test
    void addAndContains() {
        assertThat(index.contains(item1.id()), is(false));
        index.add2(item1);
        assertThat(index.contains(item1.id()), is(true));
    }

    @Test
    void returnsItems() {
        assertThat(index.items().isEmpty(), is(true));
        index.add2(item1);
        assertThat(index.items().size(), is(1));
        assertThat(index.items(), hasItems(item1));
    }

    @Test
    void removeItem() {
        index.add2(item1);

        assertThat(index.remove(item1.id(), item1.version()), is(true));

        assertThat(index.size(), is(0));
        assertThat(index.items().size(), is(0));
        assertThat(index.get(item1.id()), is(Optional.empty()));

        assertThat(index.asExactIndex().size(), is(0));
        assertThat(index.asExactIndex().items().size(), is(0));
        assertThat(index.asExactIndex().get(item1.id()), is(Optional.empty()));
    }

    @Test
    void addNewerItem() {
        TestItem newerItem = new TestItem(item1.id(), new double[]{0.f, 0.f}, item1.version() + 1);

        index.add2(item1);
        index.add2(newerItem);

        assertThat(index.size(), is(1));
        assertThat(index.get(item1.id()), is(Optional.of(newerItem)));
    }

    @Test
    void addOlderItem() {
        TestItem olderItem = new TestItem(item1.id(), new double[]{0.f, 0.f}, item1.version() - 1);

        index.add2(item1);
        index.add2(olderItem);

        assertThat(index.size(), is(1));
        assertThat(index.get(item1.id()), is(Optional.of(item1)));
    }

    @Test
    void removeUnknownItem() {
        assertThat(index.remove("foo", 0), is(false));
    }

    @Test
    void removeWithOldVersionIgnored() {
        index.add2(item1);

        assertThat(index.remove(item1.id(), item1.version() - 1), is(false));
        assertThat(index.size(), is(1));
    }

    @Test
    void findNearest() throws InterruptedException {
        index.addAll(Arrays.asList(item1, item2, item3));

        List<SearchResult<TestItem>> nearest = index.findNearest(item1.vector(), 10);

        assertThat(nearest, is(Arrays.asList(
                SearchResult.create(item1, 0d),
                SearchResult.create(item3, 0.06521261d),
                SearchResult.create(item2, 0.11621308d)
        )));
    }

    @Test
    void findReverseNeighbors() throws InterruptedException {
        index.addAll(Arrays.asList(item1, item2, item3));

        Optional<Set<TestItem>> reverseNeighbors = index.findReverseNeighbors(item1.id());
        Optional<Set<TestItem>> reverseNeighborsExact = index.asExactIndex().findReverseNeighbors(item1.id());

        assertThat(reverseNeighbors, is(reverseNeighborsExact));
    }

    @Test
    void addAllCallsProgressListener() throws InterruptedException {
        List<ProgressUpdate> updates = new ArrayList<>();

        index.addAll(Arrays.asList(item1, item2, item3), 1,
                (workDone, max) -> updates.add(new ProgressUpdate(workDone, max)), 2);

        assertThat(updates, is(Arrays.asList(
                new ProgressUpdate(2, 3),
                new ProgressUpdate(3, 3)  // emitted because its the last element
        )));
    }

    @Test
    void saveAndLoadIndex() throws IOException {
        ByteArrayOutputStream in = new ByteArrayOutputStream();

        index.add2(item1);

        index.save(in);

        HnswIndex<String, double[], TestItem> loadedIndex =
                HnswIndex.load(new ByteArrayInputStream(in.toByteArray()));

        assertThat(loadedIndex.size(), is(1));
    }
}
