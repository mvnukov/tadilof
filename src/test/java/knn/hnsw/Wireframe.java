/*******************************************************************************
 * Copyright (c) 2010-2020 Haifeng Li. All rights reserved.
 *
 * Smile is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * Smile is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Smile.  If not, see <https://www.gnu.org/licenses/>.
 ******************************************************************************/

package knn.hnsw;

import java.awt.Color;
import smile.math.MathEx;
import smile.plot.swing.Graphics;
import smile.plot.swing.Plot;

/**
 * A wire frame model specifies each edge of the physical object where two
 * mathematically continuous smooth surfaces meet, or by connecting an
 * object's constituent vertices using straight lines or curves.
 *
 * @author Haifeng Li
 */
public class Wireframe extends Plot {

    /**
     * The coordinates of vertices in the wire frame.
     */
    final double[][] vertices;
    private int[][] knn;
    private int[][] rknn;

    /**
     * Constructor.
     * @param vertices a n-by-2 or n-by-3 array which are coordinates of n vertices.
     * @param edges an m-by-2 array of which each row is the vertex indices of two
     * end points of each edge.
     */
    public Wireframe(double[][] vertices, int[][] knn, int[][] rknn) {
        super(Color.BLACK);
        this.vertices = vertices;
        this.knn = knn;
        this.rknn = rknn;
    }

    @Override
    public double[] getLowerBound() {
        return MathEx.colMin(vertices);
    }

    @Override
    public double[] getUpperBound() {
        return MathEx.colMax(vertices);
    }

    @Override
    public void paint(Graphics g) {
        g.setColor(Color.BLUE);

        for (int[] edge : knn) {
            g.drawLine(vertices[edge[0]], vertices[edge[1]]);
            g.drawPoint('O', vertices[edge[1]]);
        }

        g.setColor(Color.RED);

        for (int[] edge : rknn) {
            g.drawLine(vertices[edge[0]], vertices[edge[1]]);
            g.drawPoint(vertices[edge[1]]);
        }
    }

    /**
     * Constructor.
     * @param vertices a n-by-2 or n-by-3 array which are coordinates of n vertices.
     * @param edges an m-by-2 array of which each row is the vertex indices of two
     * end points of each edge.
     */
    public static Wireframe of(double[][] vertices, int[][] knn, int[][] rknn) {
        return new Wireframe(vertices, knn, rknn);
    }
}
