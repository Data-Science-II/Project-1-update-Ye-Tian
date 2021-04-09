
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Mustafa Nural
 *  @version 1.6
 *  @date    Sat Jan 20 16:05:52 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Quadratic Regression (quadratic terms, no cross-terms)
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
/** The `QuadRegression` class uses multiple regression to fit a quadratic
 *  surface to the data.  For example in 2D, the quadratic regression equation is
 *  <p>
 *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_0, x_0^2, x_1, x_1^2] + e
 *  <p>
 *  Has no interaction/cross-terms and adds an a constant term for intercept
 *  (must not include intercept (column of ones) in initial data matrix).
 *  @see scalation.metamodel.QuadraticFit
 *  @param x_         the initial data/input matrix (before quadratic term expansion)
 *                        must not include an intercept column of all ones
 *  @param y          the response/output vector
 *  @param fname_     the feature/variable names
 *  @param hparam     the hyper-parameters
 *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y
 */
class QuadRegression (x_ : MatriD, y: VectoD, fname_ : Strings = null, hparam: HyperParameter = null,
                      technique: RegTechnique = QR)
      extends Regression (QuadRegression.allForms (x_), y, fname_, hparam, technique)
      with ExpandableForms
{
    private val n0 = x_.dim2                                     // number of terms/columns originally
    private val nt = QuadRegression.numTerms (n0)                // number of terms/columns after expansion

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the vector 'z' into a vector of that includes additional terms,
     *  i.e., add quadratic terms.
     *  @param z  the un-expanded vector
     */
    def expand (z: VectoD): VectoD = QuadRegression.forms (z, n0, nt) 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the vector 'z', expand it and predict the response value.
     *  @param z  the un-expanded vector
     */
    def predict_ex (z: VectoD): Double = predict (expand (z))

} // QuadRegression class


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegression` companion object provides factory functions and functions
 *  for creating functional forms.
 */
object QuadRegression extends ModelFactory
{
    val drp = (null, null, QR)                                   // default remaining parameters

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `QuadRegression` object from a combined data-response matrix.
     *  @param xy         the initial combined data-response matrix (before quadratic term expansion)
     *  @param fname_     the feature/variable names
     *  @param hparam     the hyper-parameters
     *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y
     */
    def apply (xy: MatriD, fname: Strings = null, hparam: HyperParameter = null,
               technique: RegTechnique = QR): QuadRegression =
    {
        val n = xy.dim2
        if (n < 2) {
            flaw ("apply", s"dim2 = $n of the 'xy' matrix must be at least 2")
            null
        } else {
            val (x, y) = pullResponse (xy)
            new QuadRegression (x, y, fname, hparam, technique)
        } // if
    } // apply 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `QuadRegression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @see `ModelFactory`
     *  @param x          the initial data/input matrix (before quadratic term expansion)
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (use null for default)
     *  @param hparam     the hyper-parameters (use null for default)
     *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y (use OR for default)
     */
    def apply (x: MatriD, y: VectoD, fname: Strings, hparam: HyperParameter,
               technique: RegTechnique): QuadRegression =
    {
        val n = x.dim2
        if (n < 1) {
            flaw ("apply", s"dim2 = $n of the 'x' matrix must be at least 1")
            null
        } else if (rescale) {                                    // normalize the x matrix
            val (mu_x, sig_x) = (x.mean, stddev (x))
            val xn = normalize (x, (mu_x, sig_x))
            new QuadRegression (xn, y, fname, hparam, technique)
        } else {
            new QuadRegression (x, y, fname, hparam, technique)
       } // if
    } // apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The number of quadratic, linear and constant forms/terms (1, 3, 5, 7, ...).
     *  when there are no cross-terms.
     *  @param k  number of features/predictor variables (not counting intercept)
     */
    override def numTerms (k: Int) =  2 * k + 1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a vector/point 'v', compute the values for all of its quadratic,
     *  linear and constant forms/terms, returning them as a vector.
     *  No interaction/cross-terms.
     *  for 1D: v = (x_0)      => 'VectorD (1, x_0, x_0^2)'
     *  for 2D: v = (x_0, x_1) => 'VectorD (1, x_0, x_1, x_0^2, x_1^2)'
     *  @param v   the vector/point (i-th row of x) for creating forms/terms
     *  @param k   number of features/predictor variables (not counting intercept) [not used]
     *  @param nt  the number of terms
     */
    override def forms (v: VectoD, k: Int, nt: Int): VectoD =
    {
        VectorD (for (j <- 0 until nt) yield
            if (j == 0)          1.0                             // intercept term
            else if (j % 2 == 1) v(j/2)                          // linear terms
            else                 v((j-1)/2)~^2                   // quadratic terms
        ) // for
    } // forms

} // QuadRegression object

import QuadRegression._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest` object is used to test the `QuadRegression` class.
 *  > runMain scalation.analytics.QuadRegressionTest
 */
object QuadRegressionTest extends App
{
    import ExampleBPressure.{x01 => x, y}

    banner ("QuadRegression Model")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    val nTerms = numTerms (2)
    println (s"x = ${qrg.getX}")
    println (s"y = $y")

    println (s"nTerms = $nTerms")
    println (qrg.summary)

    banner ("Make Predictions")
    val z = VectorD (55.0, 102.0, 11.0, 99.0)
    val ze = qrg.expand (z)
    println (s"predict ($ze) = ${qrg.predict (ze)}")
    println (s"predict_ex ($z) = ${qrg.predict_ex (z)}")

    banner ("Forward Selection Test")
    qrg.forwardSelAll (cross = false)

    banner ("Backward Elimination Test")
    qrg.backwardElimAll (cross = false)

} // QuadRegressionTest object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest2` object is used to test the `QuadRegression` class.
 *  > runMain scalation.analytics.QuadRegressionTest2
 */
object QuadRegressionTest2 extends App
{
    import scalation.random.Normal
    import scalation.plot.Plot

    val (m, n) = (400, 1)
    val noise = new Normal (0, 10 * m * m)
    val x = new MatrixD (m, n)
    val y = new VectorD (m)
    val t = VectorD.range (0, m)

    for (i <- x.range1) { 
        x(i, 0) = i
        y(i) = i*i + i + noise.gen
    } // for

    banner ("Regression")
    val ox = VectorD.one (y.dim) +^: x
    val rg = new Regression (ox, y)
    println (rg.analyze ().report)
    println (rg.summary)
    val yp = rg.predict ()
    val e  = rg.residual

    banner ("QuadRegression")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    println (qrg.summary)
    val qyp = qrg.predict ()
    val qe  = qrg.residual

    val x0 = x.col(0)
    new Plot (x0, y, null, "y vs x")
    new Plot (t, y, yp, "y and yp vs t")
    new Plot (t, y, qyp, "y and qyp vs t")
    new Plot (x0, e, null, "e vs x")
    new Plot (x0, qe, null, "qe vs x")

} // QuadRegressionTest2 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest3` object is used to test the `QuadRegression` class.
 *  > runMain scalation.analytics.QuadRegressionTest3
 */
object QuadRegressionTest3 extends App
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
        y(k) = x(k, 0)~^2 + 2 * x(k, 1) +  noise.gen
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

    banner ("Multi-collinearity Check")
    val qx = qrg.getX
    println (corr (qx.asInstanceOf [MatrixD]))
    println (s"vif = ${qrg.vif ()}")

} // QuadRegressionTest3 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest4` object tests the `QuadRegression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It compares `Regression`, `QuadRegression` and `QuadRegression` with normalization.
 *  > runMain scalation.analytics.QuadRegressionTest4
 */
object QuadRegressionTest4 extends App
{
    import scalation.columnar_db.Relation

    banner ("auto_mpg relation")
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()

    banner ("auto_mpg (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
    println (s"x = $x")
    println (s"y = $y")
    val n  = x.dim2                                                  // number of variables including x0
    val nt = numTerms (n)                                            // number of terms for Quad

    banner ("auto_mpg: Regression")                                  // Regression
    var rg = new Regression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)
    println (s"n = $n")
   
    banner ("auto_mpg: QuadRegression")                              // QuadRegression
    rg = new QuadRegression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)
    println (s"n = $n, nt = $nt")
   
    banner ("auto_mpg: QuadRegression with normalization")           // QuadRegression with normalization
    rg = QuadRegression (x, y, drp._1, drp._2, drp._3)               // factory function normalizes data

    println (rg.analyze ().report)
    println (rg.summary)
    println (s"n = $n, nt = $nt")
   
} // QuadRegressionTest4 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest5` object tests the `QuadRegression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.QuadRegressionTest5
 */
object QuadRegressionTest5 extends App
{
    import scalation.columnar_db.Relation

    banner ("auto_mpg relation")
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()

    banner ("auto_mpg (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("auto_mpg: QuadRegression")
    val qrg = new QuadRegression (x, y)
    println (qrg.analyze ().report)
    val n  = x.dim2                                                  // number of variables
    val nt = QuadRegression.numTerms (n)                             // number of terms
    println (qrg.summary)
    println (s"n = $n, nt = $nt")
    
    banner ("Forward Selection Test")
    val (cols, rSq) = qrg.forwardSelAll ()                           // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                     // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for QuadRegression", lines = true)
    println (s"rSq = $rSq")

} // QuadRegressionTest5 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest6` object tests `QuadRegression` class using the following
 *  regression equation.
 *  <p>
 *      y = b dot x = b_0 + b_1*x1 + b_2*x_2.
 *  <p>
 *  > runMain scalation.analytics.QuadRegressionTest6
 */
object QuadRegressionTest6 extends App
{
    // 9 data points:             x1 x2  y
    val xy = new MatrixD ((9, 3), 2, 1,  2.0,               // 9-by-3 matrix
                                  3, 1,  2.5,
                                  4, 1,  3.0,
                                  2, 2,  5.0,
                                  3, 2,  5.5,
                                  4, 2,  6.0,
                                  2, 3, 10.0,
                                  3, 3, 10.5,
                                  4, 3, 11.04)

    println ("model: y = b0 + b1*x1 b2*x1^2 + b3*x2 + b4*x2^2")
    println (s"xy = $xy")

    val oxy = VectorD.one (xy.dim1) +^: xy
    val xy_ = oxy.selectCols (Array (0, 2, 3))
    println (s"xy_ = $xy_")

    banner ("SimpleRegression")
    val srg  = SimpleRegression (xy_)
    println (srg.analyze ().report)
    println (srg.summary)
    println (s"predict = ${srg.predict ()}")

    banner ("Regression")
    val rg  = Regression (oxy)
    println (rg.analyze ().report)
    println (rg.summary)
    println (s"predict = ${rg.predict ()}")

    banner ("QuadRegression")
    val qrg = QuadRegression (xy)
    println (qrg.analyze ().report)
    println (qrg.summary)
    println (s"predict = ${qrg.predict ()}")

} // QuadRegressionTest6 object


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadRegressionTest7` object compares `Regression` vs. `QuadRegression`
 *  using the following regression equations.
 *  <p>
 *      y = b dot x  = b_0 + b_1*x
 *      y = b dot x' = b_0 + b_1*x + b_2*x^2
 *  <p>
 *  > runMain scalation.analytics.QuadRegressionTest7
 */
object QuadRegressionTest7 extends App
{
    // 8 data points:             x   y
    val xy = new MatrixD ((8, 2), 1,  2,              // 8-by-2 matrix
                                  2,  5,
                                  3, 10,
                                  4, 18,
                                  5, 20,
                                  6, 30,
                                  7, 54,
                                  8, 59)

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

    val t = VectorD.range (0, xy.dim1)
    val mat = MatrixD (Seq (y, yp, yp2), false)
    println (s"mat = $mat")
    new PlotM (t, mat, null, "y vs. yp vs. yp2", true)

    banner ("Expanded Form")
    println (s"expanded x = ${qrg.getX}")
    println (s"y = ${qrg.getY}")

} // QuadRegressionTest7 object

