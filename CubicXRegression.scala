
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Mustafa Nural
 *  @version 1.6
 *  @date    Sat Jan 20 16:05:52 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Cubic Cross Regression (cubic terms, cubic cross-terms)
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
/** The `CubicXRegression` class uses multiple regression to fit a cubic with cross-terms
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
class CubicXRegression (x_ : MatriD, y: VectoD, fname_ : Strings = null, hparam: HyperParameter = null,
                       technique: RegTechnique = QR)
      extends Regression (CubicXRegression.allForms (x_), y, fname_, hparam, technique)
      with ExpandableForms
{
    private val n0 = x_.dim2                                     // number of terms/columns originally
    private val nt = CubicXRegression.numTerms (n0)              // number of terms/columns after expansion

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the vector 'z' into a vector of that includes additional terms,
     *  i.e., add quadratic terms.
     *  @param z  the un-expanded vector
     */
    def expand (z: VectoD): VectoD = CubicXRegression.forms (z, n0, nt)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the vector 'z', expand it and predict the response value.
     *  @param z  the un-expanded vector
     */
    def predict_ex (z: VectoD): Double = predict (expand (z))

} // CubicXRegression class


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegression` companion object provides methods for creating
 *  functional forms.
 */
object CubicXRegression extends ModelFactory
{
    val drp = (null, null, QR)                                   // default remaining parameters

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `CubicXRegression` object from a combined data-response matrix.
     *  @param xy         the initial combined data-response matrix (before quadratic term expansion)
     *  @param fname_     the feature/variable names
     *  @param hparam     the hyper-parameters
     *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y
     */
    def apply (xy: MatriD, fname: Strings = null, hparam: HyperParameter = null,
               technique: RegTechnique = QR): CubicXRegression =
    {
        val n = xy.dim2
        if (n < 2) {
            flaw ("apply", s"dim2 = $n of the 'xy' matrix must be at least 2")
            null
        } else {
            val (x, y) = pullResponse (xy)
            new CubicXRegression (x, y, fname, hparam, technique)
        } // if
    } // apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `CubicXRegression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @see `ModelFactory`
     *  @param x          the initial data/input matrix (before quadratic term expansion)
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (use null for default)
     *  @param hparam     the hyper-parameters (use null for default)
     *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y (use OR for default)
     */
    def apply (x: MatriD, y: VectoD, fname: Strings, hparam: HyperParameter,
               technique: RegTechnique): CubicXRegression =
    {
        val n = x.dim2
        if (n < 1) {
            flaw ("apply", s"dim2 = $n of the 'x' matrix must be at least 1")
            null
        } else if (rescale) {                                    // normalize the x matrix
            val (mu_x, sig_x) = (x.mean, stddev (x))
            val xn = normalize (x, (mu_x, sig_x))
            new CubicXRegression (xn, y, fname, hparam, technique)
        } else {
            new CubicXRegression (x, y, fname, hparam, technique)
       } // if
    } // apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The number of cubic, quadratic, linear and constant forms/terms (4, 10, 20, 35, ...).
     *  @param k  number of features/predictor variables (not counting intercept)
     */
    override def numTerms (k: Int) = (k + 1) * (k + 2) * (k + 3) / 6

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a vector/point 'p', compute the values for all of its cubic, quadratic,
     *  linear and constant forms/terms, returning them as a vector.
     *  for 1D: v = (x_0)      => 'VectorD (1, x_0, x_0^2, x_0^3)'
     *  for 2D: v = (x_0, x_1) => 'VectorD (1, x_0, x_0^2, x_0^3,
     *                                         x_1, x_1^2, x_1^3,
     *                                         x_0*x_1, x_0^2*x_1, x_0*x_1^2)'
     *  @param v   the source vector/point for creating forms/terms
     *  @param k   number of features/predictor variables (not counting intercept)
     *  @param nt  the number of terms
     */
    override def forms (v: VectoD, k: Int, nt: Int): VectoD =
    {
        val q = one (1) ++ v                     // augmented vector: [ 1., v(0), ..., v(k-1) ]
        val z = new VectorD (nt)                 // vector of all forms/terms
        var l = 0
        for (i <- 0 to k; j <- i to k; h <- j to k) { z(l) = q(i) * q(j) * q(h); l += 1 }
        z
    } // forms

} // CubicXRegression object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest` object is used to test the `CubicXRegression` class.
 *  > runMain scalation.analytics.CubicXRegressionTest
 */
object CubicXRegressionTest extends App
{
    import ExampleBPressure.{x01 => x, y}

    banner ("CubicXRegression Model")
    val crg = new CubicXRegression (x, y)
    println (crg.analyze ().report)

    val nTerms = CubicXRegression.numTerms (2)
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

} // CubicXRegressionTest object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest2` object is used to test the `CubicXRegression` class.
 *  > runMain scalation.analytics.CubicXRegressionTest2
 */
object CubicXRegressionTest2 extends App
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

    banner ("CubicXRegression")
    val crg = new CubicXRegression (x, y)
    println (crg.analyze ().report)
    println (crg.summary)

    banner ("Multi-collinearity Check")
    val rx = crg.getX
    println (corr (rx.asInstanceOf [MatrixD]))
    println (s"vif = ${crg.vif ()}")

} // CubicXRegressionTest2 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest3` object tests the `CubicXRegression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.CubicXRegressionTest3
 */
object CubicXRegressionTest3 extends App
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
    val crg = new CubicXRegression (x, y)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicXRegression.numTerms (n)                          // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Forward Selection Test")
    val (cols, rSq) = crg.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq, lines = true)
    println (s"rSq = $rSq")

} // CubicXRegressionTest3 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicXRegressionTest4` object tests the `CubicXRegression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.  Runs the cubic case.
 *  > runMain scalation.analytics.CubicXRegressionTest4
 */
object CubicXRegressionTest4 extends App
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
    val crg = new CubicXRegression (x, y, technique = QR)
//    val crg = new CubicXRegression (x, y, technique = SVD)
    println (crg.analyze ().report)

    val n  = x.dim2                                                // number of variables
    val nt = CubicXRegression.numTerms (n)                         // number of terms
    println (s"n = $n, nt = $nt")
    println (crg.summary)

    banner ("Forward Selection Test")
    val (cols, rSq) = crg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    println (s"t.dim = ${t.dim}, rSq.dim1 = ${rSq.dim1}")
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for CubicXRegression", lines = true)
    println (s"rSq = $rSq")

} // CubicXRegressionTest4 object

