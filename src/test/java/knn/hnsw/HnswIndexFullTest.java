package knn.hnsw;

import knn.*;
import org.apache.commons.math3.util.Pair;
import org.graphstream.graph.Graph;
import org.graphstream.graph.implementations.SingleGraph;
import org.graphstream.ui.view.Viewer;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import smile.read;

import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;

class HnswIndexFullTest {
    private HnswIndex<String, double[], TestItem> index;

    private int maxItemCount = 300;
    private int m = 12;
    private int efConstruction = 250;
    private int ef = 10;
    private int dimensions = 2;
    private DistanceFunction<double[], Double> distanceFunction = DistanceFunctions.DOUBLE_MANHATTAN_DISTANCE;
    private ObjectSerializer<String> itemIdSerializer = new JavaObjectSerializer<>();
    private ObjectSerializer<TestItem> itemSerializer = new JavaObjectSerializer<>();

    private TestItem item1 = new TestItem("1", new double[]{0.0110d, 0.2341d}, 10);
    private TestItem item2 = new TestItem("2", new double[]{0.2300d, 0.3891d}, 10);
    private TestItem item3 = new TestItem("3", new double[]{0.4300d, 0.9891d}, 10);

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
    void findReverseNeighbors() throws InterruptedException, InvocationTargetException {
        HnswIndex<String, double[], TestItem> index = (HnswIndex<String, double[], TestItem>) HnswIndex
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
            index.add2(new TestItem(String.valueOf(i.incrementAndGet()), tuple.toArray()));
        });

        Optional<Set<TestItem>> reverseNeighbors = index.findReverseNeighbors("55");
        System.out.println(reverseNeighbors);
        Optional<Set<TestItem>> reverseNeighborsExact = index.asExactIndex().findReverseNeighbors("55");
        System.out.println(reverseNeighborsExact);

        assertThat(getIds(reverseNeighbors), is(getIds(reverseNeighborsExact)));
    }

    @NotNull
    private Set<Integer> getIds(Optional<Set<TestItem>> reverseNeighbors) {
        return new TreeSet<>(reverseNeighbors.map(set -> set.stream().map(i -> Integer.valueOf(i.id())).collect(Collectors.toSet())).get());
    }

    public static void main(String[] args) {
        new HnswIndexFullTest().run();
    }

    public void run() {
        System.setProperty("org.graphstream.ui", "swing");
        HnswIndex<String, double[], TestItem> index = (HnswIndex<String, double[], TestItem>) HnswIndex
                .newBuilder(36, distanceFunction, 64000)
                .withCustomSerializers(itemIdSerializer, itemSerializer)
                .withM(m)
                .withEfConstruction(efConstruction)
                .withEf(ef)
                .withRemoveEnabled()
                .build();

        var dots = read.INSTANCE.csv("src/test/resources/kddcup.http.data_10_percent_corrected", ',', false, '"',
                '\\',
                null);
        AtomicInteger i = new AtomicInteger();
        Graph graph = new SingleGraph("Tutorial 1", false, true);
        Viewer viewer = graph.display(false);

        dots.stream().sequential().forEach(tuple -> {
            Node<TestItem> testItemNode = index.add2(new TestItem(String.valueOf(i.incrementAndGet()), tuple.toArray()));
            ConcurrentSkipListSet<Pair<Node<TestItem>, Double>> knn = testItemNode.getKnn();
            ConcurrentSkipListSet<Pair<Node<TestItem>, Double>> rknn = testItemNode.getRknn();

            System.out.println(testItemNode.id + "/" + testItemNode.item.id() + ": " + lof(testItemNode));
            org.graphstream.graph.Node visualNode = graph.addNode(String.valueOf(i.getAndIncrement()));
            visualNode.setAttribute("x", tuple.get(0));
            visualNode.setAttribute("y", tuple.get(1));
            visualNode.setAttribute("ui.style", "fill-color: rgb(0,100,255);");
            visualNode.setAttribute("ui.label", testItemNode.item.id() + "{" + tuple.get(0) + ":" + tuple.get(1) + "}");

        });

        Comparator<SearchResult<TestItem>> c = Comparator.comparing((sr) -> sr.item().id());

        Set<String> actualIds = index.findNeighbors("59", 10).stream().map(a -> a.item().id()).collect(Collectors.toSet());
        System.out.println(actualIds);

//        Set<String> expectedIds = index.asExactIndex().findNeighbors("59", 10).stream().map(a -> a.item().id()).collect(Collectors.toSet());
//        System.out.println(expectedIds);


        Optional<Set<TestItem>> reverseNeighbors = index.findReverseNeighbors("59");
        System.out.println(reverseNeighbors.map(s -> s.stream().map(TestItem::id).collect(Collectors.toSet())));
//        Optional<Set<TestItem>> reverseNeighborsExact = index.asExactIndex().findReverseNeighbors("59");
//        System.out.println(reverseNeighborsExact.map(s -> s.stream().map(TestItem::id).collect(Collectors.toSet())));

//        assertThat(getIds(reverseNeighbors), is(getIds(reverseNeighborsExact)));

    }

    private double lof(Node<TestItem> node) {
        ConcurrentSkipListSet<Pair<Node<TestItem>, Double>> knn = node.getKnn();
        double lrdA = localReachabilityDensity(node);
        double lrdB = 0;
        for (Pair<Node<TestItem>, Double> neighbor : knn) {
            lrdB += localReachabilityDensity(neighbor.getFirst());
        }
        return lrdB/(knn.size()*lrdA);
    }

    private double localReachabilityDensity(Node<TestItem> node) {
        ConcurrentSkipListSet<Pair<Node<TestItem>, Double>> knn = node.getKnn();
        double rD = 0;
        for (Pair<Node<TestItem>, Double> neighbor : knn) {
            rD = reachabilityDistance(node, neighbor);
        }
        return rD/knn.size();
    }

    private double reachabilityDistance(Node<TestItem> node, Pair<Node<TestItem>, Double> neighbor) {
        ConcurrentSkipListSet<Pair<Node<TestItem>, Double>> knn = neighbor.getFirst().getKnn();
        Pair<Node<TestItem>, Double> last = knn.first();

        return Math.max(last.getSecond(), neighbor.getSecond());
    }
}
