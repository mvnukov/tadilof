import smile.*;
import smile.data.DataFrame
import smile.math.kernel.GaussianKernel
import smile.plot.swing.*
import smile.projection.kpca
import smile.projection.pca
import java.awt.Color


fun main() {
    val iris = read.arff("src/main/resources/iris.arff");
    val setosa = DataFrame.of(iris.stream().filter { row -> row.getString("class").equals("Iris-setosa") });
    val versicolor = DataFrame.of(iris.stream().filter { row -> row.getString("class").equals("Iris-versicolor") });
    val virginica = DataFrame.of(iris.stream().filter { row -> row.getString("class").equals("Iris-virginica") });

//    val pca = kpca(iris.toArray(), GaussianKernel(45.0), 3);
    val pca = pca(iris.toArray());

    val x2 = pca.project(iris.toArray());
    PlotGrid.splom(DataFrame.of(x2).merge(iris.select("class")), '*', "class").window();

//    draw("setosa1", pca(setosa), "V1", "V4");
//    draw("setosa2", pca(setosa), "V2", "V4");
//    draw("setosa3", pca(setosa), "V3", "V4");
//    draw("setosa4", pca(setosa), "V4", "V4");

}

private fun pca(setosa: DataFrame): DataFrame {
    val pca = pca(setosa.toArray())
    pca.setProjection(4)
    val project = pca.project(setosa.toArray())
    return DataFrame.of(project).merge(setosa)
}

private fun draw(title: String, iris: DataFrame, x: String, y: String) {
//    var canvas = ScatterPlot.of(iris, x, y, "class", '*').canvas();
    val doubles = iris.select(x).toMatrix().transpose().toArray()[0]
    val canvas = LinePlot.of(doubles, Line.Style.DOT).canvas();
    canvas.title = title
    canvas.setAxisLabels("x", "y");
    canvas.window();
}