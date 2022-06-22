package knn.hnsw;

import knn.*;
import org.apache.commons.math3.util.Pair;
import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.implementations.SingleGraph;
import org.graphstream.ui.view.Viewer;
import smile.read;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

public class HnswTest {
    private HnswIndex<String, double[], TestItem> index;

    private int maxItemCount = 300;
    private int m = 12;
    private int efConstruction = 250;
    private int ef = 2;
    private int dimensions = 2;
    private DistanceFunction<double[], Double> distanceFunction = DistanceFunctions.DOUBLE_MANHATTAN_DISTANCE;
    private ObjectSerializer<String> itemIdSerializer = new JavaObjectSerializer<>();
    private ObjectSerializer<TestItem> itemSerializer = new JavaObjectSerializer<>();

    private TestItem item1 = new TestItem("1", new double[]{0.0110d, 0.2341d}, 10);
    private TestItem item2 = new TestItem("2", new double[]{0.2300d, 0.3891d}, 10);
    private TestItem item3 = new TestItem("3", new double[]{0.4300d, 0.9891d}, 10);

    public static void main(String[] args) throws InterruptedException, InvocationTargetException {
        System.setProperty("org.graphstream.ui", "swing");
        new HnswTest().run();

    }

    private void run() throws InvocationTargetException {
        HnswIndex<String, double[], TestItem> index = (HnswIndex<String, double[], TestItem>) HnswIndex
                .newBuilder(2, distanceFunction, 64000)
                .withCustomSerializers(itemIdSerializer, itemSerializer)
                .withM(m)
                .withEfConstruction(efConstruction)
                .withEf(ef)
                .withRemoveEnabled()
                .build();

        var dots = read.INSTANCE.csv("src/test/resources/kddcup.http.data_1_percent_corrected", ',', false, '"', '\\',
                null);
        AtomicInteger i = new AtomicInteger();
        Graph graph = new SingleGraph("Tutorial 1", false, true);
        Viewer viewer = graph.display(false);

        dots.stream().sequential().forEach(tuple -> {
            index.add2(new TestItem(String.valueOf(i.get()), tuple.toArray()));

            org.graphstream.graph.Node visualNode = graph.addNode(String.valueOf(i.getAndIncrement()));
            visualNode.setAttribute("x", tuple.get(0));
            visualNode.setAttribute("y", tuple.get(1));
            visualNode.setAttribute("ui.style", "fill-color: rgb(0,100,255);");
//
//            for (int j = 0; j < index.size(); j++) {
//                Node<TestItem> node = index.nodes.get(j);
//
//                if (node != null) {
//
//                    for (int k = 0; k < node.getConnections(0).size(); k++) {
//                        Edge edge = graph.addEdge("X" + j + "Y" + k, String.valueOf(node.id),
//                                String.valueOf(node.getConnections(0).get(k)), true);
////                        edge.setAttribute("ui.style", "fill-color: rgb(255,100,0);");
//                    }
//                }
//            }
////            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                throw new RuntimeException(e);
//            }
        });

        Optional<Set<TestItem>> reverseNeighbors = index.findReverseNeighbors("1");
        System.out.println(reverseNeighbors);
        Optional<Set<TestItem>> reverseNeighborsExact = index.asExactIndex().findReverseNeighbors("1");
        System.out.println(reverseNeighborsExact);

        var knn = new ArrayList<int[]>();
        var rknn = new ArrayList<int[]>();

//        ViewPanel view = (ViewPanel) viewer.getDefaultView(); // ViewPanel is the view for gs-ui-swing
//        view.resizeFrame(800, 600);
//        view.getCamera().setViewCenter(3000, 8000, 0);
//        view.getCamera().setViewPercent(0.25);

        for (int j = 0; j < index.nodes.length(); j++) {
            Node<TestItem> node = index.nodes.get(j);

            if (node != null) {
                for (Pair<Node<TestItem>, Double> k : node.getKnn()) {
                    Edge edge = graph.addEdge("X" + j + "Y" + k, String.valueOf(node.id),
                            String.valueOf(k.getKey().id), true);
                }
            }

//            node.getRknn().forEach(rknnId -> {
//                rknn.add(new int[]{rknnId, node.id});
//            });
        }
    }
//
//        int[][] knnArray = knn.toArray(new int[knn.size()][]);
//        int[][] rknnArray = rknn.toArray(new int[rknn.size()][]);

}

