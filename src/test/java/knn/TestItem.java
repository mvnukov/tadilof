package knn;

import java.util.Arrays;

public class TestItem implements Item<String, double[]> {

    private static final long serialVersionUID = 1L;

    private final String id;
    private final double[] vector;
    private final long version;

    public TestItem(String id, double[] vector) {
        this(id, vector, 0);
    }

    public TestItem(String id, double[] vector, long version) {
        this.id = id;
        this.vector = vector;
        this.version = version;
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public double[] vector() {
        return vector;
    }

    @Override
    public long version() {
        return version;
    }

    @Override
    public int dimensions() {
        return vector.length;
    }

    @Override
    public String toString() {
        return "TestItem{" +
                "id='" + id + '\'' +
                ", vector=" + Arrays.toString(vector) +
                ", version=" + version +
                '}';
    }
}