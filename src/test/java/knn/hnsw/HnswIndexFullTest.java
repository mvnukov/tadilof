package knn.hnsw;

import knn.*;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import smile.plot.swing.Wireframe;
import smile.read;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;

class HnswIndexFullTest {
    private HnswIndex<String, float[], TestItem, Float> index;

    private int maxItemCount = 300;
    private int m = 12;
    private int efConstruction = 250;
    private int ef = 10;
    private int dimensions = 2;
    private DistanceFunction<float[], Float> distanceFunction = DistanceFunctions.FLOAT_MANHATTAN_DISTANCE;
    private ObjectSerializer<String> itemIdSerializer = new JavaObjectSerializer<>();
    private ObjectSerializer<TestItem> itemSerializer = new JavaObjectSerializer<>();

    private TestItem item1 = new TestItem("1", new float[]{0.0110f, 0.2341f}, 10);
    private TestItem item2 = new TestItem("2", new float[]{0.2300f, 0.3891f}, 10);
    private TestItem item3 = new TestItem("3", new float[]{0.4300f, 0.9891f}, 10);

    @BeforeEach
    void setUp() {
        HnswIndex<String, float[], TestItem, Float> build = (HnswIndex<String, float[], TestItem, Float>) HnswIndex
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
    void findReverseNeighbors() throws InterruptedException, InvocationTargetException {
        HnswIndex<String, float[], knn.TestItem, Float> index = (HnswIndex<String, float[], knn.TestItem, Float>) HnswIndex
                .newBuilder(21, distanceFunction, 64000)
                .withCustomSerializers(itemIdSerializer, itemSerializer)
                .withM(m)
                .withEfConstruction(efConstruction)
                .withEf(ef)
                .withRemoveEnabled()
                .build();

        var dots = read.INSTANCE.csv("src/test/resources/kddcup.http.data_1_percent_26_corrected", ',', false, '"',
                '\\',
                null);
        AtomicInteger i = new AtomicInteger();
        dots.stream().forEach(tuple -> {
            index.add(new TestItem(String.valueOf(i.incrementAndGet()), toFloats(tuple.toArray())));
        });

        Optional<Set<TestItem>> reverseNeighbors = index.findReverseNeighbors("1");
        System.out.println(reverseNeighbors);
        Optional<Set<TestItem>> reverseNeighborsExact = index.asExactIndex().findReverseNeighbors("1");
        System.out.println(reverseNeighborsExact);

        assertThat(getIds(reverseNeighbors), is(getIds(reverseNeighborsExact)));
    }

    @NotNull
    private Set<Integer> getIds(Optional<Set<TestItem>> reverseNeighbors) {
        return new TreeSet<>(reverseNeighbors.map(set -> set.stream().map(i -> Integer.valueOf(i.id())).collect(Collectors.toSet())).get());
    }

    @Test
    void findNeighbors() throws InterruptedException {
        HnswIndex<String, float[], knn.TestItem, Float> index = (HnswIndex<String, float[], knn.TestItem, Float>) HnswIndex
                .newBuilder(37, distanceFunction, 64000)
                .withCustomSerializers(itemIdSerializer, itemSerializer)
                .withM(m)
                .withEfConstruction(efConstruction)
                .withEf(ef)
                .withRemoveEnabled()
                .build();

        var dots = read.INSTANCE.csv("src/test/resources/kddcup.http.data_10_percent_corrected", ',', false, '"', '\\',
                null);
        AtomicInteger i = new AtomicInteger();
        dots.stream().forEach(tuple -> {
            index.add(new TestItem(String.valueOf(i.incrementAndGet()), toFloats(tuple.toArray())));
        });

        Comparator<SearchResult<TestItem, Float>> c = Comparator.comparing((sr)->sr.item().id());
        List<SearchResult<TestItem, Float>> nearest =
                index.findNeighbors("1000", 10);
        List<SearchResult<TestItem, Float>> nearestExact = index.asExactIndex().findNeighbors("1000", 10);

        nearest.sort(c);
        nearestExact.sort(c);
        assertThat(nearest, is(nearestExact));
    }

    private float[] toFloats(double[] doubles) {
        float[] floats = new float[doubles.length];
        for (int i = 0; i < doubles.length; i++) {
            floats[i] = (float) doubles[i];
        }
        return floats;
    }
}
