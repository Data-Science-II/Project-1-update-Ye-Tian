
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 1.6
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model Framework: Predictor for Matrix Input
 */

package scalation.analytics

import scala.collection.mutable.{Map, Set}
import scala.math.{abs, log, pow, sqrt}
import scala.util.control.Breaks.{break, breakable}

import scalation.linalgebra._
import scalation.math.{noDouble, sq}
import scalation.plot.Plot
import scalation.stat.Statistic
import scalation.stat.StatVector.corr
import scalation.random.CDF.studentTCDF
import scalation.random.PermutedVecI
import scalation.util.{banner, time}
import scalation.util.Unicode.sub

import Fit._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PredictorMat` abstract class supports multiple predictor analytics,
 *  such as `Regression`.
 *  In this case, 'x' is multi-dimensional [1, x_1, ... x_k].  Fit the parameter
 *  vector 'b' in for example the regression equation
 *  <p>
 *      y  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  <p>
 *  Note, "protected val" arguments required by `ResponseSurface`.
 *  @param x       the input/data m-by-n matrix
 *                     (augment with a first column of ones to include intercept in model)
 *  @param y       the response/output m-vector
 *  @param fname   the feature/variable names (if null, use x_j's)
 *  @param hparam  the hyper-parameters for the model
 */
abstract class PredictorMat (protected val x: MatriD, protected val y: VectoD,
                             protected var fname: Strings, hparam: HyperParameter)
         extends Fit (y, x.dim2, x.dim2 - 1, x.dim1 - x.dim2) with Predictor
         // if not using an intercept df = (x.dim2, x.dim1-x.dim2), correct by calling 'resetDF' method from `Fit`
{
    if (x.dim1 != y.dim) flaw ("constructor", "row dimensions of x and y are incompatible")
    if (x.dim1 <= x.dim2) {
        flaw ("constructor", s"PredictorMat requires more rows ${x.dim1} than columns ${x.dim2}")
    } // if

    private   val DEBUG   = true                                         // debug flag
    private   val DEBUG2  = false                                        // verbose debug flag
    protected val m       = x.dim1                                       // number of data points (rows)
    protected val n       = x.dim2                                       // number of parameter (columns)
    protected val k       = x.dim2 - 1                                   // number of variables (k = n-1) - assumes intercept
    private   val stream  = 0                                            // random number stream to use
    private   val permGen = PermutedVecI (VectorI.range (0, m), stream)  // permutation generator

    protected var b: VectoD = null                                       // parameter/coefficient vector [b_0, b_1, ... b_k]
    protected var e: VectoD = null                                       // residual/error vector [e_0, e_1, ... e_m-1]

    if (fname == null) fname = x.range2.map ("x" + _).toArray            // default feature/variable names

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'used' data matrix 'x'.  Mainly for derived classes where 'x' is expanded
     *  from the given columns in 'x_', e.g., `QuadRegression` add squared columns.
     */
    def getX: MatriD = x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'used' response vector 'y'.  Mainly for derived classes where 'y' is
     *  transformed, e.g., `TranRegression`, `Regression4TS`.
     */
    def getY: VectoD = y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a predictive model 'y_ = f(x_) + e' where 'x_' is the data/input
     *  matrix and 'y_' is the response/output vector.  These arguments default
     *  to the full dataset 'x' and 'y', but may be restricted to a training
     *  dataset.  Training involves estimating the model parameters 'b'.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    def train (x_ : MatriD = x, y_ : VectoD = y): PredictorMat

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a predictive model 'y_ = f(x_) + e' where 'x_' is the data/input
     *  matrix and 'y_' is the response/output vector.  These arguments default
     *  to the full dataset 'x' and 'y', but may be restricted to a training
     *  dataset.  Training involves estimating the model parameters 'b'.
     *  The 'train2' method should work like the 'train' method, but should also
     *  optimize hyper-parameters (e.g., shrinkage or learning rate).
     *  Only implementing classes needing this capability should implement this method.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    def train2 (x_ : MatriD = x, y_ : VectoD = y): PredictorMat =
    {
        throw new UnsupportedOperationException ("train2: not supported - no hyper-parameters to optimize")
    } // train2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the error (difference between actual and predicted) and useful
     *  diagnostics for the test dataset.
     *  @param x_e  the test/full data/input matrix (defualts to full x)
     *  @param y_e  the test/full response/output vector (defualts to full y)
     */
    def eval (x_e: MatriD = x, y_e: VectoD = y): PredictorMat =
    {
        val yp = predict (x_e)                                           // y predicted for x_e (test/full)
        e = y_e - yp                                                     // compute residual/error vector e
        diagnose (e, y_e, yp)                                            // compute diagnostics
        this
    } // eval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the error (difference between actual and predicted) and useful
     *  diagnostics for the test dataset.  Requires predicted responses to be
     *  passed in.
     *  @param ym   the training/full mean actual response/output vector
     *  @param y_e  the test/full actual response/output vector
     *  @param yp   the test/full predicted response/output vector
     */
    def eval (ym: Double, y_e: VectoD, yp: VectoD): PredictorMat = ???

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Analyze a dataset using this model using ordinary training with the
     *  'train' method.
     *  @param x_   the training/full data/input matrix
     *  @param y_   the training/full response/output vector
     *  @param x_e  the test/full data/input matrix
     *  @param y_e  the test/full response/output vector
     */
    def analyze (x_ : MatriD = x, y_ : VectoD = y,
                 x_e: MatriD = x, y_e: VectoD = y): PredictorMat =
    {
        train (x_, y_)                                                   // train the model on the training dataset
//      val ym = y_.mean                                                 // compute mean of training response - FIX use ym
        eval (x_e, y_e)                                                  // evaluate using the testing dataset
        this
    } // analyze

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the hyper-parameters.
     */
    def hparameter: HyperParameter = hparam

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of parameter/coefficient values.
     */
    def parameter: VectoD = b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a basic report on the trained model.
     *  @see 'summary' method for more details
     */
    def report: String =
    {
        s"""
REPORT
    hparameter hp  = $hparameter
    parameter  b   = $parameter
    fitMap     qof = $fitMap
        """
    } // report

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute and return summary diagostics for the regression model.
     */
    def summary: String =
    {
        (if (fname != null) "fname = " + fname.deep else "") + 
        super.summary (b, {
            val facCho = new Fac_Cholesky (x.t * x)                      // create a Cholesky factorization
            val l_inv  = facCho.factor1 ().inverse                       // take inverse of l from Cholesky factorization
            val varCov = l_inv.t * l_inv * mse_                          // variance-covariance matrix
            varCov.getDiag ().map (sqrt (_)) },                          // standard error of coefficients
            vif ())                                                      // Variance Inflation Factors (VIFs)
    } // summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of residuals/errors.
     */
    def residual: VectoD = e

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of 'y = f(z)' by evaluating the formula 'y = b dot z',
     *  e.g., '(b_0, b_1, b_2) dot (1, z_1, z_2)'.
     *  @param z  the new vector to predict
     */
    def predict (z: VectoD): Double = b dot z

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of 'y = f(z)' by evaluating the formula 'y = b dot z',
     *  for each row of matrix 'z'.
     *  @param z  the new matrix to predict
     */
    def predict (z: MatriD = x): VectoD = VectorD (for (i <- z.range1) yield predict (z(i)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    def buildModel (x_cols: MatriD): PredictorMat

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform forward selection to find the most predictive variable to add the
     *  existing model, returning the variable to add and the new model.
     *  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols     the columns of matrix x currently included in the existing model
     *  @param index_q  index of Quality of Fit (QoF) to use for comparing quality
     */
    def forwardSel (cols: Set [Int], index_q: Int = index_rSqBar): (Int, PredictorMat) =
    {
        var j_mx   = -1                                                  // best column, so far
        var mod_mx = null.asInstanceOf [PredictorMat]                    // best model, so far
        var fit_mx = noDouble                                            // best fit, so far

        for (j <- x.range2 if ! (cols contains j)) {
            val cols_j = cols + j                                        // try adding variable/column x_j
            val x_cols = x.selectCols (cols_j.toArray)                   // x projected onto cols_j columns
            val mod_j   = buildModel (x_cols)                            // regress with x_j added
            mod_j.train ().eval ()                                       // train model, evaluate QoF
            val fit_j = mod_j.fit(index_q)                               // new fit
            if (fit_j > fit_mx) { j_mx = j; mod_mx = mod_j; fit_mx = fit_j }
        } // for
        if (j_mx == -1) {
            flaw ("forwardSel", "could not find a variable x_j to add: j = -1")
        } // if
        (j_mx, mod_mx)                                                    // return best column and model
    } // forwardSel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform forward selection to find the most predictive variables to have
     *  in the model, returning the variables added and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param index_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param cross    whether to include the cross-validation QoF measure
     */
    def forwardSelAll (index_q: Int = index_rSqBar, cross: Boolean = true): (Set [Int], MatriD) =
    {
        val rSq  = new MatrixD (x.dim2 - 1, 3)                           // R^2, R^2 Bar, R^2 cv
        val cols = Set (0)                                               // start with x_0 in model

        println (s"forwardSelAll (l = 0): cols = $cols")
        breakable { for (l <- 0 until x.dim2 - 1) {
            val (j, mod_j) = forwardSel (cols)                           // add most predictive variable
            if (j == -1) break
            cols     += j                                                // add variable x_j
            val fit_j = mod_j.fit
            rSq(l)    = if (cross) Fit.qofVector (fit_j, mod_j.crossValidate ())   // use new model, mod_j, with cross
                        else       Fit.qofVector (fit_j, null)                     // use new model, mod_j, no cross
            if (DEBUG) {
                val k = cols.size - 1
                println (s"==> forwardSelAll (l = $l): add (#$k) variable $j, cols = $cols, qof = ${fit_j(index_q)}")
            } // if
        }} // breakable for

        (cols, rSq.slice (0, cols.size-1))
    } // forwardSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform backward elimination to find the least predictive variable to remove
     *  from the existing model, returning the variable to eliminate, the new parameter
     *  vector and the new Quality of Fit (QoF).  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols     the columns of matrix x currently included in the existing model
     *  @param index_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param first    first variable to consider for elimination
     *                        (default (1) assume intercept x_0 will be in any model)
     */
    def backwardElim (cols: Set [Int], index_q: Int = index_rSqBar, first: Int = 1): (Int, PredictorMat) =
    {
        var j_mx   = -1                                                  // best column, so far
        var mod_mx = null.asInstanceOf [PredictorMat]                    // best model, so far
        var fit_mx = noDouble                                            // best fit, so far

        for (j <- first until x.dim2 if cols contains j) {
            val cols_j = cols - j                                        // try removing variable/column x_j
            val x_cols = x.selectCols (cols_j.toArray)                   // x projected onto cols_j columns
            val mod_j  = buildModel (x_cols)                             // regress with x_j added
            mod_j.train ().eval ()                                       // train model, evaluate QoF
            val fit_j = mod_j.fit(index_q)                               // new fit
            if (fit_j > fit_mx) { j_mx = j; mod_mx = mod_j; fit_mx = fit_j }
        } // for
        if (j_mx == -1) {
            flaw ("backwardElim", "could not find a variable x_j to eliminate: j = -1")
        } // if
        (j_mx, mod_mx)                                                   // return best column and model
    } // backwardElim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform backward elimination to find the least predictive variables to remove
     *  from the full model, returning the variables left and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param index_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param first    first variable to consider for elimination
     *  @param cross    whether to include the cross-validation QoF measure
     */
    def backwardElimAll (index_q: Int = index_rSqBar, first: Int = 1, cross: Boolean = true): (Set [Int], MatriD) =
    {
        val rSq  = new MatrixD (x.dim2 - 1, 3)                           // R^2, R^2 Bar, R^2 cv
        val cols = Set (Array.range (0, x.dim2) :_*)                     // start with all x_j in model

        println (s"backwardElimAll (l = 0): cols = $cols")
        breakable { for (l <- 1 until x.dim2 - 1) {
            val (j, mod_j) = backwardElim (cols, first)                  // remove most predictive variable
            if (j == -1) break
            cols     -= j                                                // remove variable x_j
            val fit_j = mod_j.fit
            rSq(l)    = if (cross) Fit.qofVector (fit_j, mod_j.crossValidate ())   // use new model, mod_j, with cross
                        else       Fit.qofVector (fit_j, null)                     // use new model, mod_j, no cross
            if (DEBUG) {
                println (s"<== backwardElimAll (l = $l): remove (#$l) variable $j, cols = $cols, qof = ${fit_j(index_q)}")
            } // if
        }} // breakable for

//      (cols, rSq.slice (0, cols.size-1))
        (cols, reverse (rSq.slice (1, rSq.dim1)))
    } // backwardElimAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform stepwise regression to find the most predictive variables to have
     *  in the model, returning the variables left and the new Quality of Fit (QoF)
     *  measures for all steps.  At each step it calls 'forwardSel' and 'backwardElim'
     *  and takes the best of the two actions.  Stops when neither action yields improvement.
     *  @see `Fit` for index of QoF measures.
     *  @param index_q  index of Quality of Fit (QoF) to use for comparing quality
     *  @param cross    whether to include the cross-validation QoF measure
     */
    def stepRegressionAll (index_q: Int = index_rSqBar, cross: Boolean = true): (Set [Int], MatriD) =
    {
        val rSq    = new MatrixD (x.dim2 - 1, 3)                         // R^2, R^2 Bar, R^2 cv
        val cols   = Set (0)                                             // start with x_0 in model
        var last_q = -1.0

        println (s"stepRegressionAll (l = 0): cols = $cols")
        breakable { for (l <- 1 until x.dim2 - 1) {
            val (j, mod_j) = forwardSel (cols)                           // add most predictive variable OR
            val (k, mod_k) = backwardElim (cols, 1)                      // remove most predictive variable
            if (j+k == -2) break                                         // both are -1 (no change is possible)
            val fit_j  = if (j == -1) mod_k.fit else mod_j.fit           // best addition is j
            val fit_k  = if (k == -1) mod_j.fit else mod_k.fit           // best removal is k
            val fit_jq = fit_j(index_q)                                  // QoF for best addition
            val fit_kq = fit_k(index_q)                                  // QoF for best removal

            if (math.max(fit_jq, fit_kq) < last_q) {
                break
            } else if (fit_jq >= fit_kq) {
                cols  += j                                                          // add variable x_j
                rSq(l) = if (cross) Fit.qofVector (fit_j, mod_j.crossValidate ())   // use new model, mod_j, with cross
                         else       Fit.qofVector (fit_j, null)                     // use new model, mod_j, no cross
                if (DEBUG) {
                    println (s"<== stepRegressionAll (l = $l): add (#$l) variable $j, cols = $cols, qof = ${fit_j(index_q)}")
                } // if
                last_q = fit_jq
            } else {
                cols  -= k                                                          // remove variable x_k
                rSq(l) = if (cross) Fit.qofVector (fit_k, mod_j.crossValidate ())   // use new model, mod_j, with cross
                         else       Fit.qofVector (fit_k, null)                     // use new model, mod_j, no cross
                if (DEBUG) {
                    println (s"<== stepRegressionAll (l = $l): remove (#$l) variable $k, cols = $cols, qof = ${fit_k(index_q)}")
                } // if
                last_q = fit_kq
            } // if
        }} // breakable for

        (cols, rSq.slice (0, cols.size-1))
    } // stepRegressionAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a matrix that is in reverse row order of the given matrix 'a'.
     *  @param a  the given matrix
     */
    def reverse (a: MatriD): MatriD =
    {
        val b = new MatrixD (a.dim1, a.dim2)
        for (i <- a.range1) b(i) = a(a.dim1 - 1 - i)
        b
    } // reverse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the correlation matrix for the columns in data matrix 'xx'.
     *  @param xx  the data matrix shose correlation matrix is sought
     */
    override def corrMatrix (xx: MatriD = x): MatriD = corr (xx.asInstanceOf [MatrixD])    // FIX

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Variance Inflation Factor 'VIF' for each variable to test
     *  for multi-collinearity by regressing 'x_j' against the rest of the variables.
     *  A VIF over 10 indicates that over 90% of the variance of 'x_j' can be predicted
     *  from the other variables, so 'x_j' may be a candidate for removal from the model.
     *  Note:  override this method to use a superior regression technique.
     *  @param skip  the number of columns of x at the beginning to skip in computing VIF
     */
    def vif (skip: Int = 1): VectoD =
    {
        val vifV = new VectorD (x.dim2 - skip)                         // VIF vector for x columns except skip columns
        for (j <- skip until x.dim2) {
            val (x_noj, x_j) = pullResponse (x, j)                     // x matrix without column j, only column j
            val rg_j  = new Regression (x_noj, x_j)                    // regress with x_j removed
            rg_j.analyze ()                                            // train model, evaluate QoF
            val rSq_j = rg_j.fit(index_rSq)                            // R^2 for predicting x_j
//          if (rSq_j.isNaN) diagnoseMat (x_noj)
            if (DEBUG2) println (s"vif: for variable x_$j, rSq_$j = $rSq_j")
            vifV(j-1) =  1.0 / (1.0 - rSq_j)                           // store vif for x_1 in vifV(0)
        } // for
        vifV
    } // vif

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use 'k'-fold cross-validation to compute test Quality of Fit (QoF) measures
     *  by iteratively dividing the dataset into a test dataset and a training dataset.
     *  Each test dataset is defined by 'idx' and the rest of the data is the training dataset.
     *  FIX - problem with forward selection
     *  @see 'showQofStatTable' in `Fit` object for printing the returned 'stats'.
     *  @param k      the number of cross-validation iterations/folds (defaults to 10x).
     *  @param rando  flag indicating whether to use randomized or simple cross-validation
     */
    def crossValidate (k: Int = 10, rando: Boolean = true): Array [Statistic] =
    {
        if (k < MIN_FOLDS) flaw ("crossValidate", s"k = $k must be at least $MIN_FOLDS")
        val stats   = qofStatTable                                       // create table for QoF measures
        val indices = if (rando) permGen.igen.split (k)                  // k groups of indices
                      else       VectorI (0 until m).split (k)

        for (idx <- indices) {
            val (x_e, x_) = x.splitRows (idx)                            // test, training data/input matrices
            val (y_e, y_) = y.split (idx)                                // test, training response/output vectors

            train (x_, y_)                                               // train model on the training dataset
            eval (x_e, y_e)                                              // evaluate model on the test dataset
            val qof = fit                                                // get Quality of Fit (QoF) measures
            if (DEBUG) println (s"crossValidate: fitMap = $fitMap")
            if (qof(index_sst) > 0.0) {                                  // requires variation in test set
                for (q <- qof.range) stats(q).tally (qof(q))             // tally these QoF measures
            } // if
        } // for

        stats
    } // crossValidate

} // PredictorMat abstract class


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PredictorMat` companion object provides a meythod for splitting
 *  a combined data matrix in predictor matrix and a response vector.
 */
object PredictorMat
{
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Analyze a dataset using the given model where training includes
     *  hyper-parameter optimization with the 'train2' method.
     *  @param model  the model to be used
     */
    def analyze2 (model: PredictorMat)
    {
        println (model.train2 ().eval ().report)
    } // analyze2

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the model on the full dataset (i.e., train and evaluate on full dataset).
     *  Calls 'analyze2' which includes hyper-parameter optimization .
     *  @param modelName  the name of the model being tested
     *  @param model      the model to be used
     *  @param doPlot     whether to plot the actual vs. predicted response
     */
    def test2 (modelName: String, model: PredictorMat, doPlot: Boolean = true)
    {
        banner (s"Test $modelName")
        val (x, y) = (model.getX, model.getY)                            // get full data x and response y
        analyze2 (model)                                                 // train and evalaute the model on full dataset
        if (doPlot) {
            val idx = VectorD.range (0, y.dim)                           // data instance index (for horizonal axis)
            val yp  = model.predict (x)                                  // predicted response
            new Plot (idx, y, yp, s"$modelName: y vs. yp", true)         // plot actual vs. predicted response
        } // if
    } // test2

} // PredictorMat object


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PredictorMatTest` is used to test the `PredictorMat` abstract class
 *  and its derived classes using the `ExampleBasketBall` dataset containing
 *  data matrix 'x' and response vector 'y'.
 *  Shift imports for `ExampleAutoMPG` dataset.
 *  > runMain scalation.analytics.PredictorMatTest
 */
object PredictorMatTest extends App
{
    import ActivationFun._
    import PredictorMat.test2

//  import ExampleBasketBall._
    import ExampleAutoMPG._

    println ("xy = " + xy)                                       // combined data-response matrix

    val xs = ox.sliceCol (0, 2)                                  // only the first two columns of ox
    val xr = x.sliceCol (0, 1)                                   // only the first column of x

    val mod0 = new NullModel (y)
    mod0.test ("NullModel")                                      // 1

    var mod = null.asInstanceOf [PredictorMat]

    mod = new SimplerRegression (xr, y, fname)
    mod.test ("SimplerRegression")                               // 2

    mod = new SimpleRegression (xs, y, fname)
    mod.test ("SimpleRegression")                                // 3

    mod = new Regression (x, y, fname)
    mod.test ("Regression with no intercept")                    // 4

    mod = new Regression (ox, y, fname)
    mod.test ("Regression with intercept")                       // 5

    mod = new Regression_WLS (ox, y, fname)
    mod.test ("Regression_WLS with intercept")                   // 6

    mod = new RidgeRegression (x, y, fname)
    mod.test ("RidgeRegression with no intercept")               // 7

    mod = new RidgeRegression (ox, y, fname)
    mod.test ("RidgeRegression with intercept")                  // 8

    mod = new LassoRegression (x, y, fname)
    mod.test ("LassoRegression with no intercept")               // 9

    mod = new LassoRegression (ox, y, fname)
    mod.test ("LassoRegression with intercept")                  // 10

    mod = new TranRegression (ox, y, fname)
    mod.test ("TranRegression with intercept - log")             // 11

    mod = new TranRegression (ox, y, fname, null, sqrt _, sq _)
    mod.test ("TranRegression with intercept - sqrt")            // 12

    mod = TranRegression (ox, y, fname)
    mod.test ("TranRegression with intercept - box-cox")         // 13

    mod = new QuadRegression (x, y, fname)
    mod.test ("QuadRegression")                                  // 14

    mod = new QuadXRegression (x, y, fname)
    mod.test ("QuadXRegression")                                 // 15

    mod = new CubicRegression (x, y, fname)
    mod.test ("CubicRegression")                                 // 16

    mod = new CubicXRegression (x, y, fname)
    mod.test ("CubicXRegression")                                // 17

    mod = new PolyRegression (xr, y, 4, fname)
    mod.test ("PolyRegression")                                  // 18

    mod = new PolyORegression (xr, y, 4, fname)
    mod.test ("PolyORegression")                                 // 19

    mod = new TrigRegression (xr, y, 8, fname)
    mod.test ("TrigRegression")                                  // 20

    mod = new ExpRegression (ox, y, fname)
    mod.test ("ExpRegression")                                   // 21

    mod = new PoissonRegression (ox, y, fname)
    mod.test ("PoissonRegression")                               // 22 - FIX

    mod = new KNN_Predictor (x, y, fname)
    mod.test ("KNN_Predictor")                                   // 23 

    mod = new RegressionTree (x, y, fname)
    mod.test ("RegressionTree")                                  // 24

    mod = new RegressionTree_GB (x, y, fname)
    mod.test ("RegressionTree_GB")                               // 25

    mod = ELM_3L1 (xy, -1, fname)                                // use factory function for rescaling
    mod.test ("ELM_3L1")                                         // 26

    mod = QuadELM_3L1 (xy, -1, fname)                            // use factory function for rescaling
    mod.test ("QuadELM_3L1")                                     // 27

    mod = QuadXELM_3L1 (xy, -1, fname)                           // use factory function for rescaling
    mod.test ("QuadXELM_3L1")                                    // 28

    mod = Perceptron (oxy, fname)
    test2 ("Perceptron with sigmoid", mod)                       // 29

    mod = Perceptron (oxy, fname, f = f_tanh)
    test2 ("Perceptron with tanh", mod)                          // 30 

    mod = Perceptron (oxy, fname, Optimizer.hp.updateReturn ("eta", 0.001), f = f_id)
    test2 ("Perceptron with id", mod)                            // 31

    mod = Perceptron (oxy, fname, f = f_lreLU)
    test2 ("Perceptron with lreLU", mod)                         // 32

/*
    banner ("Test Perceptron with eLU")
    mod = Perceptron (oxy, fname, f0 = f_eLU)
    test2 ("Perceptron with eLU", mod)                           // 33 - FIX
*/

} // PredictorMatTest

