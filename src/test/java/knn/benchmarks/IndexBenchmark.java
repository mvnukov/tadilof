package knn.benchmarks;

import knn.*;
import knn.hnsw.HnswIndex;
import org.openjdk.jmh.annotations.*;

import java.util.Random;

@State(Scope.Benchmark)
public class IndexBenchmark {
    private static final Random random = new Random();

    @Param({"20", "40", "100"})
    public int DIMENSIONS = 40;
    private HnswIndex<String, double[], TestItem> index;
    private static long iteration;
    private TestItem testItem;

    @Setup(Level.Invocation)
    public void createPoint() {
        double[] doubles = new double[DIMENSIONS];
        for (int i = 0; i < DIMENSIONS; i++) {
            doubles[i] = random.nextFloat();
        }
        testItem = new TestItem(String.valueOf(iteration++), doubles);
    }

    @Setup(Level.Trial)
    public void buildHnsw() {
        index = HnswIndex
                .newBuilder(DIMENSIONS, DistanceFunctions.DOUBLE_MANHATTAN_DISTANCE, 640000)
                .withCustomSerializers(new JavaObjectSerializer<String>(), new JavaObjectSerializer<TestItem>())
                .withM(12)
                .withEfConstruction(250)
                .withEf(10)
                .withRemoveEnabled()
                .build();

        System.out.println("Built an index");
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        System.out.printf("Finished with rows: %d and columns: %d%n", index.size(), DIMENSIONS);
    }


    @Benchmark
    public void index(IndexBenchmark indexBenchmark) {
        indexBenchmark.index.add2(indexBenchmark.testItem);
    }


}
