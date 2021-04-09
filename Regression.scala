
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 1.6
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Multiple Linear Regression (linear terms, no cross-terms)
 */

package scalation.analytics

import scala.collection.mutable.Set
import scala.util.control.Breaks.{break, breakable}

import scalation.linalgebra._
import scalation.math.noDouble
import scalation.plot.{Plot, PlotM}
import scalation.random.CDF.studentTCDF
import scalation.stat.Statistic
import scalation.stat.StatVector.corr
import scalation.util.banner
import scalation.util.Unicode.sub

import Fit._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegTechnique` object defines the implementation techniques available.
 */
object RegTechnique extends Enumeration
{
    type RegTechnique = Value
    val QR, Cholesky, SVD, LU, Inverse = Value
    val techniques = Array (QR, Cholesky, SVD, LU, Inverse)
   
} // RegTechnique

import MatrixTransform._
import RegTechnique._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Regression` class supports multiple linear regression.  In this case,
 *  'x' is multi-dimensional [1, x_1, ... x_k].  Fit the parameter vector 'b' in
 *  the regression equation
 *  <p>
 *      y  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  <p>
 *  where 'e' represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve the parameter vector 'b'
 *  using the Normal Equations:
 *  <p>
 *      x.t * x * b  =  x.t * y 
 *      b  =  fac.solve (.)
 *  <p>
 *  Five factorization techniques are provided:
 *  <p>
 *      'QR'         // QR Factorization: slower, more stable (default)
 *      'Cholesky'   // Cholesky Factorization: faster, less stable (reasonable choice)
 *      'SVD'        // Singular Value Decomposition: slowest, most robust
 *      'LU'         // LU Factorization: better than Inverse
 *      'Inverse'    // Inverse/Gaussian Elimination, classical textbook technique
 *  <p>
 *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  Note, not intended for use when the number of degrees of freedom 'df' is negative.
 *  @see en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)
 *  @param x          the data/input m-by-n matrix
 *                        (augment with a first column of ones to include intercept in model)
 *  @param y          the response/output m-vector
 *  @param fname_     the feature/variable names
 *  @param hparam     the hyper-parameters (it doesn't have any, but may be used by derived classes)
 *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y
 */
class Regression (x: MatriD, y: VectoD,
                  fname_ : Strings = null, hparam: HyperParameter = null,
                  technique: RegTechnique = QR)
      extends PredictorMat (x, y, fname_, hparam)
{
    private val DEBUG = true                                             // debug flag

    type Fac_QR = Fac_QR_H [MatriD]                                      // change as needed (H => Householder)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a solver for the Normal Equations using the selected factorization technique.
     *  @param x_  the matrix to be used by the solver
     */
    private def solver (x_ : MatriD): Factorization =
    {
        technique match {                                                // select the factorization technique
        case QR       => new Fac_QR (x_, false)                          // QR Factorization
        case Cholesky => new Fac_Cholesky (x_.t * x_)                    // Cholesky Factorization
        case SVD      => new SVD (x_)                                    // Singular Value Decomposition
        case LU       => new Fac_LU (x_.t * x_)                          // LU Factorization
        case _        => new Fac_Inv (x_.t * x_)                         // Inverse Factorization
        } // match
    } // solver

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  multiple regression equation
     *  <p>
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_1 , ... x_k] + e
     *  <p>
     *  using the ordinary least squares 'OLS' method.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatriD = x, y_ : VectoD = y): Regression =
    {
        val fac = solver (x_)                                            // create selected factorization technique
        fac.factor ()                                                    // factor the matrix, either X or X.t * X
        b = technique match {                                            // solve for parameter/coefficient vector b
            case QR       => fac.solve (y_)                              // R * b = Q.t * y
            case Cholesky => fac.solve (x_.t * y_)                       // L * L.t * b = X.t * y
            case SVD      => fac.solve (y_)                              // b = V * Σ^-1 * U.t * y
            case LU       => fac.solve (x_.t * y_)                       // b = (X.t * X) \ X.t * y
            case _        => fac.solve (x_.t * y_)                       // b = (X.t * X)^-1 * X.t * y
        } // match
        if (b(0).isNaN) flaw ("train", s"parameter b = $b")
        if (DEBUG) (s"train: parameter b = $b")
        this
    } // train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the error (difference between actual and predicted) and useful
     *  diagnostics for the test dataset.  Overridden for efficiency.
     *  @param x_e  the test/full data/input matrix
     *  @param y_e  the test/full response/output vector
     */
    override def eval (x_e: MatriD = x, y_e: VectoD = y): Regression =
    {
        val yp = x_e * b                                                 // y predicted for x_e (test/full)
        e = y_e - yp                                                     // compute residual/error vector e
        diagnose (e, y_e, yp)                                            // compute diagnostics
        this
    } // eval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of 'y = f(z)' by evaluating the formula 'y = Z b',
     *  @param z  the new matrix to predict
     */
    override def predict (z: MatriD = x): VectoD = z * b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    def buildModel (x_cols: MatriD): Regression =
    {
        new Regression (x_cols, y, null, null, technique)
    } // buildModel

} // Regression class


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Regression` companion object provides factory apply functions and a testing method.
 */
object Regression extends ModelFactory
{
    val drp = (null, null, QR)                                   // default remaining parameters

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Regression` object from a combined data-response matrix.
     *  The last column is assumed to be the response column.
     *  @param xy         the combined data-response matrix (predictors and response)
     *  @param fname      the feature/variable names
     *  @param hparam     the hyper-parameters
     *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y
     */
    def apply (xy: MatriD, fname: Strings = null, hparam: HyperParameter = null,
               technique: RegTechnique = QR): Regression = 
    {
        val n = xy.dim2
        if (n < 2) {
            flaw ("apply", s"dim2 = $n of the 'xy' matrix must be at least 2")
            null
        } else {
            val (x, y) = pullResponse (xy)
            new Regression (x, y, fname, hparam, technique)
       } // if
    } // apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Regression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @see `ModelFactory`
     *  @param x          the data/input m-by-n matrix
     *                        (augment with a first column of ones to include intercept in model)
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (use null for default)
     *  @param hparam     the hyper-parameters (use null for default)
     *  @param technique  the technique used to solve for b in x.t*x*b = x.t*y (use OR for default)
     */
    def apply (x: MatriD, y: VectoD, fname: Strings, hparam: HyperParameter,
               technique: RegTechnique): Regression =
    {
        val n = x.dim2
        if (n < 1) {
            flaw ("apply", s"dim2 = $n of the 'x' matrix must be at least 1")
            null
        } else if (rescale) {                                   // normalize the x matrix
            val (mu_x, sig_x) = (x.mean, stddev (x))
            val xn = normalize (x, (mu_x, sig_x))
            new Regression (xn, y, fname, hparam, technique)
        } else {
            new Regression (x, y, fname, hparam, technique)
       } // if
    } // apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the various regression techniques.
     *  @param x      the data/input matrix
     *  @param y      the response/output vector
     *  @param z      a vector to predict
     *  @param fname  the names of features/variable
     */
    def test (x: MatriD, y: VectoD, z: VectoD, fname: Strings = null)
    {
        println (s"x = $x")
        println (s"y = $y")

        for (tec <- techniques) {                                      // use 'tec' Factorization
            banner (s"Fit the parameter vector b using $tec")
            val rg = new Regression (x, y, fname, null, tec)           // use null for hyper-parameters
            println (rg.analyze ().report)
            println (rg.summary)

            val yp = rg.predict (x)                                    // predict y for several points
            println (s"predict (x) = $yp")
            new Plot (y, yp, null, tec.toString, true)

            val yp1 = rg.predict (z)                                   // predict y for one point
            println (s"predict ($z) = $yp1")
        } // for
    } // test

} // Regression object

import Regression.test

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest` object tests `Regression` class using the following
 *  regression equation.
 *  <p>
 *      y  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2.
 *  <p>
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  > runMain scalation.analytics.RegressionTest
 */
object RegressionTest extends App
{
    // 5 data points: constant term, x_1 coordinate, x_2 coordinate
    val x = new MatrixD ((5, 3), 1.0, 36.0,  66.0,                     // 5-by-3 matrix
                                 1.0, 37.0,  68.0,
                                 1.0, 47.0,  64.0,
                                 1.0, 32.0,  53.0,
                                 1.0,  1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)
    val z = VectorD (1.0, 20.0, 80.0)

//  println ("model: y = b_0 + b_1*x_1 + b_2*x_2")
    println ("model: y = b₀ + b₁*x₁ + b₂*x₂")

    test (x, y, z)

} // RegressionTest object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest2` object tests `Regression` class using the following
 *  regression equation, which has a perfect fit.
 *  <p>
 *      y = b dot x = b_0 + b_1*x1 + b_2*x_2.
 *  <p>
 *  > runMain scalation.analytics.RegressionTest2
 */
object RegressionTest2 extends App
{
    // 4 data points: constant term, x_1 coordinate, x_2 coordinate
    val x = new MatrixD ((4, 3), 1.0, 1.0, 1.0,                        // 4-by-3 matrix
                                 1.0, 1.0, 2.0,
                                 1.0, 2.0, 1.0,
                                 1.0, 2.0, 2.0)
    val y = VectorD (6.0, 8.0, 7.0, 9.0)
    val z = VectorD (1.0, 2.0, 3.0)

//  println ("model: y = b_0 + b_1*x1 + b_2*x_2")
    println ("model: y = b₀ + b₁*x₁ + b₂*x₂")

    test (x, y, z)

} // RegressionTest2 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest3` object tests the multi-collinearity method in the
 *  `Regression` class using the following regression equation on the Blood
 *  Pressure dataset.  Also performs Collinearity Diagnostics.
 *  <p>
 *      y = b dot x = b_0 + b_1*x_1 + b_2*x_2 + b_3*x_3 + b_4 * x_4
 *  <p>
 *  @see online.stat.psu.edu/online/development/stat501/12multicollinearity/05multico_vif.html
 *  > runMain scalation.analytics.RegressionTest3
 */
object RegressionTest3 extends App
{
    import ExampleBPressure._

//  println ("model: y = b_0 + b_1*x1 + b_2*x_ + b3*x3 + b4*x42")
    println ("model: y = b₀ + b₁∙x₁ + b₂∙x₂ + b₃∙x₃ + b₄∙x₄")

    val z = VectorD (1.0, 46.0, 97.5, 7.0, 95.0)

//  test (x, y, z, fname)                                            // no intercept
    test (ox, y, z, ofname)                                          // with intercept

    banner ("Collinearity Diagnostics")
    println ("corr (x) = " + corr (x))                               // correlations of column vectors in x

} // RegressionTest3 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest4` object tests the multi-collinearity method in the
 *  `Regression` class using the following regression equation on the Blood
 *  Pressure dataset.  It also applies forward selection and backward elimination.
 *  <p>
 *      y = b dot x = b_0 + b_1*x_1 + b_2*x_2 + b_3*x_3 + b_4 * x_4
 *  <p>
 *  @see online.stat.psu.edu/online/development/stat501/12multicollinearity/05multico_vif.html
 *  @see online.stat.psu.edu/online/development/stat501/data/bloodpress.txt
 *  > runMain scalation.analytics.RegressionTest4
 */
object RegressionTest4 extends App
{
    import ExampleBPressure._
    val x = ox                                                       // use ox for intercept

//  println ("model: y = b_0 + b_1*x1 + b_2*x_ + b3*x3 + b4*x42")
    println ("model: y = b₀ + b₁∙x₁ + b₂∙x₂ + b₃∙x₃ + b₄∙x₄")
    println ("x = " + x)
    println ("y = " + y)

    banner ("Parameter Estimation and Quality of Fit")
    val rg = new Regression (x, y, ofname)
    println (rg.analyze ().report)
    println (rg.summary)

    banner ("Collinearity Diagnostics")
    println ("corr (x) = " + corr (x))                               // correlations of column vectors in x

    banner ("Multi-collinearity Diagnostics")
    println ("vif      = " + rg.vif ())                              // test multi-colinearity (VIF)

    banner ("Forward Selection Test")
    rg.forwardSelAll (cross = false)

    banner ("Backward Elimination Test")
    rg.backwardElimAll (cross = false)

} // RegressionTest4 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest5` object tests the cross-validation for the `Regression`
 *  class using the following regression equation on the Blood Pressure dataset.
 *  <p>
 *      y = b dot x = b_0 + b_1*x_1 + b_2*x_2 + b_3*x_3 + b_4 * x_4
 *  <p>
 *  > runMain scalation.analytics.RegressionTest5
 */
object RegressionTest5 extends App
{
    import ExampleBPressure._
    val x = ox                                                       // use ox for intercept

//  println ("model: y = b_0 + b_1*x1 + b_2*x_ + b3*x3 + b4*x42")
    println ("model: y = b₀ + b₁∙x₁ + b₂∙x₂ + b₃∙x₃ + b₄∙x₄")
    println ("x = " + x)
    println ("y = " + y)

    val rg  = new Regression (x, y, fname)
    banner ("Cross-Validation")
    val stats = rg.crossValidate ()
    showQofStatTable (stats)

} // RegressionTest5 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest6` object tests `Regression` class using the following
 *  regression equation.
 *  <p>
 *      y = b dot x = b_0 + b_1*x1 + b_2*x_2.
 *  <p>
 *  > runMain scalation.analytics.RegressionTest6
 */
object RegressionTest6 extends App
{
    // 7 data points: constant term, x_1 coordinate, x_2 coordinate
    val x = new MatrixD ((7, 3), 1.0, 1.0, 1.0,                      // 7-by-3 matrix
                                 1.0, 1.0, 2.0,
                                 1.0, 2.0, 1.0,
                                 1.0, 2.0, 2.0,
                                 1.0, 2.0, 3.0,
                                 1.0, 3.0, 2.0,
                                 1.0, 3.0, 3.0)
    val y = VectorD (6.0, 8.0, 9.0, 11.0, 13.0, 13.0, 16.0)
    val z = VectorD (1.0, 1.0, 3.0)

//  println ("model: y = b_0 + b_1*x1 + b_2*x_2")
    println ("model: y = b₀ + b₁*x₁ + b₂*x₂")
    println ("x = " + x)
    println ("y = " + y)

    test (x, y, z)

} // RegressionTest6 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest7` object tests `Regression` class using the following
 *  regression equation.
 *  <p>
 *      y = b dot x = b_0 + b_1*x1 + b_2*x_2.
 *  <p>
 *  > runMain scalation.analytics.RegressionTest7
 */
object RegressionTest7 extends App
{
    val xy = new MatrixD ((9, 4), 1, 0, 0, 0,
                                  1, 0, 1, 0,
                                  1, 0, 2, 0,
                                  1, 1, 0, 0,
                                  1, 1, 1, 1,
                                  1, 1, 2, 1,
                                  1, 2, 0, 0,
                                  1, 2, 1, 1,
                                  1, 2, 2, 1)
    val rg = Regression (xy)
    println (rg.analyze ().report)
    println (rg.summary)

} // RegressionTest7 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest8` object tests the `Regression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  > runMain scalation.analytics.RegressionTest8
 */
object RegressionTest8 extends App
{
    import scalation.columnar_db.Relation

    banner ("auto_mpg relation")
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()

    banner ("auto_mpg (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
    val m = y.dim
    println (s"x = $x")
    println (s"y = $y")

    banner ("auto_mpg regression")
//  val rg = new Regression (x, y)                                // without interecept
    val rg = new Regression (VectorD.one (m) +^: x, y)            // with intercept
    println (rg.analyze ().report)
    println (rg.summary)

    banner ("auto_mpg cross-validation")
    val stats = rg.crossValidate ()
    showQofStatTable (stats)

} // RegressionTest8 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest9` object is used to test the `Regression` class.
 *  <p>
 *      y = b dot x = b0 + b1 * x1
 *  <p>
 *  This version uses gradient descent to search for the optimal solution for 'b'.
 *  Try normalizing the data first.
 *  @see `MatrixTransform`
 *  > runMain scalation.analytics.RegressionTest9
 */
object RegressionTest9 extends App
{
    import scala.util.control.Breaks.{breakable, break}
    import MatrixTransform.golden
    import ExampleBPressure._
    val x = ox                                                        // use ox for intercept

    val stop = new StoppingRule ()

    println ("x = " + x)

    val ITER = 100                                                    // number of iterations/epochs
    val eta  = 0.000006                                               // try different values for the learning rate (g)
//  val eta  = 0.00012                                                // try different values for the learning rate (gg)
    val rg   = new Regression (x, y)                                  // create a regression model, don't train
    val mu   = y.mean                                                 // mean of y
    val sst  = (y dot y) - mu * mu * x.dim1                           // sum of squares total
    var b    = new VectorD (x.dim2)                                   // starting point [0, 0] for parameter vector b

    banner (s"Regression Model: gradient descent: eta = $eta")
    breakable { for (it <- 1 to ITER) {
        val yp = x * b                                                // y predicted
        val e  = y - yp                                               // error
        val g  = x.t * e                                              // - gradient
        val gg = golden (g)                                           // - golden gradient
        println (s"g = $g, gg = $gg")
        b     += g * eta                                              // update parameter b
//      b     += gg * eta                                             // update parameter b (golden gradient)
        val sse = e dot e                                             // sum of squares error
        val rSq = 1.0 - sse / sst                                     // coefficient of determination
        println (s"for iteration $it, b = $b, sse = $sse, rSq = $rSq")
        val (b_best, sse_best) = stop.stopWhen (b, sse) 
        if (b_best != null) {
            val rSq_best = 1.0 - sse_best / sst                       // best coefficient of determination
            println (s"solution b_best = $b_best, sse_best = $sse_best, rSq_best = $rSq_best")
            break
        } // if
    }} // breakable for

} // RegressionTest9 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest10` object tests the `Regression` class using the following
 *  regression equation.
 *  <p>
 *      y = b dot x = b_0 + b_1*x1 + b_2*x_2.
 *  <p>
 *  Show effects of increasing collinearity.
 *  > runMain scalation.analytics.RegressionTest10
 */
object RegressionTest10 extends App
{
//                               c x1 x2
    val x = new MatrixD ((4, 3), 1, 1, 1,
                                 1, 2, 2,
                                 1, 3, 3,
                                 1, 4, 0)                             // change 0 by .5 to 4

    val y = VectorD (1, 3, 3, 4)

    val v = x.sliceCol (0, 2)
    banner (s"Test without column x2")
    println (s"v = $v")
    var rg = new Regression (v, y)
    println (rg.analyze ().report)
    println (rg.summary)

    for (i <- 0 to 8) {
        banner (s"Test Increasing Collinearity: x_32 = ${x(3, 2)}")
        println (s"x = $x")
        println (s"corr (x) = ${corr (x)}")
        rg = new Regression (x, y)
        println (rg.analyze ().report)
        println (rg.summary)
        x(3, 2) += 0.5
   } // for

} // RegressionTest10 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest11` object tests the `Regression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RegressionTest11
 */
object RegressionTest11 extends App
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
    val rg = new Regression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)

//  val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
    val (cols, rSq) = rg.backwardElimAll ()                        // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, rSq.dim1)                            // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Regression", lines = true)

    println (s"rSq = $rSq")

} // RegressionTest11 object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTest12` object tests the `Regression` class to illustrate
 *  projection of 6 vectors/points onto a plane in 3D space, see the last
 *  exercise in Regression section of the textbook.
 *  <p>
 *      y = b dot x = b_1*x1 + b_2*x_2.
 *  <p>
 *  > runMain scalation.analytics.RegressionTest12
 */
object RegressionTest12 extends App
{
    val xy = new MatrixD ((6, 3), 1, 1, 2.8,
                                  1, 2, 4.2,
                                  1, 3, 4.8,
                                  2, 1, 5.3,
                                  2, 2, 5.5,
                                  2, 3, 6.5)
    val rg = Regression (xy)
    println (rg.analyze ().report)
    println (rg.summary)

} // RegressionTest12 object

