
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Mustafa Nural
 *  @version 1.6
 *  @date    Sat Jan 20 16:05:52 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Cubic Regression (cubic terms, quadratic cross-terms)
 */

package scalation.analytics

import scala.collection.mutable.Set

import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD}
import scalation.linalgebra.VectorD.one
import scalation.math.double_exp
import scalation.plot.PlotM
import scalation.stat.Statistic
import scalation.util.banner

import MatrixTransform._
import RegTechnique._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegression` class uses multiple regression to fit a cubic with cross-terms
 *  surface to the data.  For example in 2D, the cubic cross-terms regression equation is
 *  <p>
 *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_0, x_0^2, x_0^3,
                                                     x_1, x_1^2, x_1^3,
                                                     x_0*x_1, x_0^2*x_1, x_0*x_1^2] + e
 *  <p>
 *  Adds an a constant term for intercept (must not include intercept (column of ones)
 *  in initial data matrix).
 *  @see scalation.metamodel.QuadraticFit
 *  @param x_         the input vectors/points
 *  @param y          the response vector
 *  @param fname_     the feature/variable names
 *  @param hparam     the hyper-parameters
 *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y
 */
class CubicRegression (x_ : MatriD, y: VectoD, fname_ : Strings = null, hparam: HyperParameter = null,
                       technique: RegTechnique = QR)
      extends Regression (CubicRegression.allForms (x_), y, fname_, hparam, technique)
      with ExpandableForms
{
    private val n0 = x_.dim2                                     // number of terms/columns originally
    private val nt = CubicRegression.numTerms (n0)               // number of terms/columns after expansion

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the vector 'z' into a vector of that includes additional terms,
     *  i.e., add quadratic terms.
     *  @param z  the un-expanded vector
     */
    def expand (z: VectoD): VectoD = CubicRegression.forms (z, n0, nt)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the vector 'z', expand it and predict the response value.
     *  @param z  the un-expanded vector
     */
    def predict_ex (z: VectoD): Double = predict (expand (z))

} // CubicRegression class


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegression` companion object provides methods for creating
 *  functional forms.
 */
object CubicRegression extends ModelFactory
{
    val drp = (null, null, QR)                                   // default remaining parameters

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `CubicRegression` object from a combined data-response matrix.
     *  @param xy         the initial combined data-response matrix (before quadratic term expansion)
     *  @param fname_     the feature/variable names
     *  @param hparam     the hyper-parameters
     *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y
     */
    def apply (xy: MatriD, fname: Strings = null, hparam: HyperParameter = null,
               technique: RegTechnique = QR): CubicRegression =
    {
        val n = xy.dim2
        if (n < 2) {
            flaw ("apply", s"dim2 = $n of the 'xy' matrix must be at least 2")
            null
        } else {
            val (x, y) = pullResponse (xy)
            new CubicRegression (x, y, fname, hparam, technique)
        } // if
    } // apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `CubicRegression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @see `ModelFactory`
     *  @param x          the initial data/input matrix (before quadratic term expansion)
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (use null for default)
     *  @param hparam     the hyper-parameters (use null for default)
     *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y (use OR for default)
     */
    def apply (x: MatriD, y: VectoD, fname: Strings, hparam: HyperParameter,
               technique: RegTechnique): CubicRegression =
    {
        val n = x.dim2
        if (n < 1) {
            flaw ("apply", s"dim2 = $n of the 'x' matrix must be at least 1")
            null
        } else if (rescale) {                                    // normalize the x matrix
            val (mu_x, sig_x) = (x.mean, stddev (x))
            val xn = normalize (x, (mu_x, sig_x))
            new CubicRegression (xn, y, fname, hparam, technique)
        } else {
            new CubicRegression (x, y, fname, hparam, technique)
       } // if
    } // apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The number of cubic, quadratic, linear and constant forms/terms (4, 8, 13, 19, ...).
     *  @param k  number of features/predictor variables (not counting intercept)
     */
    override def numTerms (k: Int) = (k * (k + 5) + 2) / 2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a vector/point 'p', compute the values for all of its cubic, quadratic,
     *  linear and constant forms/terms, returning them as a vector.
     *  for 1D: v = (x_0)      => 'VectorD (1, x_0, x_0^2, x_0^3)'
     *  for 2D: v = (x_0, x_1) => 'VectorD (1, x_0, x_0^2, x_0^3,
     *                                         x_1, x_1^2, x_1^3,
     *                                         x_0*x_1)'
     *  @param v   the source vector/point for creating forms/terms
     *  @param k   number of features/predictor variables (not counting intercept)
     *  @param nt  the number of terms
     */
    override def forms (v: VectoD, k: Int, nt: Int): VectoD =
    {
        val z = new VectorD (k)
        for (i <- z.range) z(i) = v(i) ~^3
        QuadXRegression.forms (v, k, QuadXRegression.numTerms (k)) ++ z
    } // forms

} // CubicRegression object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest` object is used to test the `CubicRegression` class.
 *  > runMain scalation.analytics.CubicRegressionTest
 */
object CubicRegressionTest extends App
{
    import ExampleBPressure.{x01 => x, y}

    banner ("CubicRegression Model")
    val crg = new CubicRegression (x, y)
    println (crg.analyze ().report)

    val nTerms = CubicRegression.numTerms (2)
    println (s"x = ${crg.getX}")
    println (s"y = $y")

    println ("nTerms    = " + nTerms)
    println (crg.summary)

    banner ("Make Predictions")
    val z = VectorD (55.0, 102.0, 11.0, 99.0)
    val ze = crg.expand (z)
    println (s"predict ($ze) = ${crg.predict (ze)}")
    println (s"predict_ex ($z) = ${crg.predict_ex (z)}")

    banner ("Forward Selection Test")
    crg.forwardSelAll (cross = false)

    banner ("Backward Elimination Test")
    crg.backwardElimAll (cross = false)

    banner ("Stepwise Regression Test")
    crg.stepRegressionAll (cross = false)

} // CubicRegressionTest object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest2` object is used to test the `CubicRegression` class.
 *  > runMain scalation.as4360.htmlnalytics.CubicRegressionTest2
 */
object CubicRegressionTest2 extends App
{
    import scalation.random.Normal
    import scalation.stat.StatVector.corr

    val s      = 20
    val grid   = 1 to s
    val (m, n) = (s*s, 2)
    val noise  = new Normal (0, 10 * s * s)
    val x = new MatrixD (m, n)
    val y = new VectorD (m)

    var k = 0
    for (i <- grid; j <- grid) {
        x(k) = VectorD (i, j)
        y(k) = x(k, 0)~^2 + 2 * x(k, 1) + x(k, 0) * x(k, 1) +  noise.gen
        k += 1
    } // for

    banner ("Regression")
    val ox = VectorD.one (y.dim) +^: x
    val rg = new Regression (ox, y)
    println (rg.analyze ().report)
    println (rg.summary)

    banner ("QuadRegression")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    println (qrg.summary)

    banner ("CubicRegression")
    val crg = new CubicRegression (x, y)
    println (crg.analyze ().report)
    println (crg.summary)

    banner ("Multi-collinearity Check")
    val rx = crg.getX
    println (corr (rx.asInstanceOf [MatrixD]))
    println (s"vif = ${crg.vif ()}")

} // CubicRegressionTest2 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest3` object tests the `CubicRegression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicRegressionTest3
 */
object CubicRegressionTest3 extends App
{
    import scalation.columnar_db.Relation

    banner ("auto_mpg relation")
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()

    banner ("auto_mpg (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)

//  import ExampleAutoMPG.{x, y}

    println (s"x = $x")
    println (s"y = $y")

    banner ("auto_mpg regression")
    val crg = new CubicRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Forward Selection Test")
    val (cols, rSq) = crg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv
    val k = cols.size - 1
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, lines = true)
    println (s"k = $k, nt = $nt")
    println (s"rSq = $rSq")

} // CubicRegressionTest3 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest4` object tests the `CubicRegression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.  Runs the cubic case.
 *  > runMain scalation.analytics.CubicRegressionTest4
 */
object CubicRegressionTest4 extends App
{
    import scalation.columnar_db.Relation

    banner ("auto_mpg relation")
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()

    banner ("auto_mpg (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("auto_mpg regression")
//    val crg = new CubicRegression (x, y, technique = QR)
    val crg = new CubicRegression (x, y, technique = SVD)
    println (crg.analyze ().report)

    val n  = x.dim2                                               // number of variables
    val nt = CubicRegression.numTerms (n)                         // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Forward Selection Test")
    val (cols, rSq) = crg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv
    val k = cols.size - 1
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for CubicRegression", lines = true)
    println (s"k = $k, nt = $nt")
    println (s"rSq = $rSq")

} // CubicRegressionTest4 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicRegressionTest5` object compares `Regression` vs. `QuadRegression`
 *  vs. CubicRegression.
 *  using the following regression equations.
 *  <p>
 *      y = b dot x  = b_0 + b_1*x
 *      y = b dot x' = b_0 + b_1*x + b_2*x^2
 *      y = b dot x' = b_0 + b_1*x + b_2*x^2 + b_3*x^3
 *  <p>
 *  > runMain scalation.analytics.CubicRegressionTest5
 */
object CubicRegressionTest5 extends App
{
    // 8 data points:             x   y
    val xy = new MatrixD ((8, 2), 1,  2,              // 8-by-2 matrix
                                  2, 11,
                                  3, 25,
                                  4, 28,
                                  5, 30,
                                  6, 26,
                                  7, 42,
                                  8, 60)

    println ("model: y = b0 + b1*x1 + b2*x1^2")
    println (s"xy = $xy")

    val y   = xy.col(1)
    val oxy = VectorD.one (xy.dim1) +^: xy

    banner ("Regression")
    val rg  = Regression (oxy)                        // create a regression model
    println (rg.analyze ().report)                    // analyze and report
    println (rg.summary)                              // show summary
    val yp = rg.predict ()                            // y predicted for Regression
    println (s"predict = $yp")

    banner ("QuadRegression")
    val qrg = QuadRegression (xy)                     // create a quadratic regression model
    println (qrg.analyze ().report)                   // analyze and report
    println (qrg.summary)                             // show summary
    val yp2 = qrg.predict ()                          // y predicted for Regression
    println (s"predict = $yp2")

    banner ("CubicRegression")
    val crg = CubicRegression (xy)                    // create a cubic regression model
    println (crg.analyze ().report)                   // analyze and report
    println (crg.summary)                             // show summary
    val yp3 = crg.predict ()                          // y predicted for Regression
    println (s"predict = $yp2")

    val t = VectorD.range (0, xy.dim1)
    val mat = MatrixD (Seq (y, yp, yp2, yp3), false)
    println (s"mat = $mat")
    new PlotM (t, mat, null, "y vs. yp vs. yp2 vs. yp3", true)

    banner ("Expanded Form")
    println (s"expanded x = ${crg.getX}")
    println (s"y = ${crg.getY}")

} // CubicRegressionTest5 object

