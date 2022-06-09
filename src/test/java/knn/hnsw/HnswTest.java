package knn.hnsw;

import knn.*;
import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.implementations.SingleGraph;
import org.graphstream.ui.swing_viewer.ViewPanel;
import org.graphstream.ui.view.Viewer;
import smile.read;

import java.awt.*;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

public class HnswTest {
    private HnswIndex<String, float[], TestItem, Float> index;

    private int maxItemCount = 300;
    private int m = 12;
    private int efConstruction = 250;
    private int ef = 2;
    private int dimensions = 2;
    private DistanceFunction<float[], Float> distanceFunction = DistanceFunctions.FLOAT_MANHATTAN_DISTANCE;
    private ObjectSerializer<String> itemIdSerializer = new JavaObjectSerializer<>();
    private ObjectSerializer<TestItem> itemSerializer = new JavaObjectSerializer<>();

    private TestItem item1 = new TestItem("1", new float[]{0.0110f, 0.2341f}, 10);
    private TestItem item2 = new TestItem("2", new float[]{0.2300f, 0.3891f}, 10);
    private TestItem item3 = new TestItem("3", new float[]{0.4300f, 0.9891f}, 10);

    public static void main(String[] args) throws InterruptedException, InvocationTargetException {
        System.setProperty("org.graphstream.ui", "swing");
        new HnswTest().run();

    }

    private void run() throws InterruptedException, InvocationTargetException {
        HnswIndex<String, float[], knn.TestItem, Float> index = (HnswIndex<String, float[], knn.TestItem, Float>) HnswIndex
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
            index.add(new TestItem(String.valueOf(i.get()), toFloats(tuple.toArray())));

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
                for (int k = 0; k < node.getConnections(0).size(); k++) {
                    knn.add(new int[]{node.id, node.getConnections(0).get(k)});
                    Edge edge = graph.addEdge("X" + j + "Y" + k, String.valueOf(node.id),
                            String.valueOf(node.getConnections(0).get(k)), true);
                }

                node.reverseConnections.forEach(rknnId -> {
                    rknn.add(new int[]{rknnId, node.id});
                });
            }
        }
//
//        int[][] knnArray = knn.toArray(new int[knn.size()][]);
//        int[][] rknnArray = rknn.toArray(new int[rknn.size()][]);

    }

    private float[] toFloats(double[] doubles) {
        float[] floats = new float[doubles.length];
        for (int i = 0; i < doubles.length; i++) {
            floats[i] = (float) doubles[i];
        }
        return floats;
    }

}
