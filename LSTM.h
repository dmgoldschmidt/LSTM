#ifndef LSTM_H
#define LSTM_H
#include <iostream>
#include <cassert>
#include <cmath>
#include <fenv.h>
#include "util.h"
#include "GetOpt.h"
#include "Matrix.h"
#include "Array.h"

using namespace std;

ColVector<double> augment(ColVector<double> x){
   ColVector<double> y(x.nrows() + 1);
   y[0] = 1.0;
   for(int i = 0;i < x.nrows();i++) y[i+1] = x[i];
   return y;
 }

double sigma(double u){return 1/(1+exp(-u));}

ColVector<double>& squash(ColVector<double>& x){
  for(int i = 0;i < x.nrows();i++) x[i] = sigma(x[i]);
  return x;
}

ColVector<double>& bulge(ColVector<double> x){
  for(int i = 0;i < x.nrows();i++) x[i] =2*sigma(x[i])-1;
  return x;
}

struct Gate {
  /* A gate is just the function sigma(g(v,s,x)) where 
   * sigma(g) = 1/(1+exp(-g)), g(v,s,x) is the affine function 
   * g = W_v*v + W_s*s + W_x*x + b, where v = previous cell readout, 
   * s =  previous cell state, and x = current cell input.
   * In the special case gate_no = 0, there is no s input and sigma
   * is replaced by 2*sigma - 1. 
   * It is called a gate because its output modulates some information 
   * signal z by component-wise multiplication.  So a gate component 
   * near 1 lets most of the corresponding component of z pass through, 
   * while a value near zero blocks most of that component.
   
   * The gate also operates in back-propagation mode where input and 
   * output are reversed.  In this mode, the input is dE/dg where E is 
   * some sort of error signal.  The gate then computes d(sigma)/dg,
   * dg/dW_i (i = 0,1,2) and dg/dz_i (i = 0,1)which then multiply dE/dg 
   * to get dE/dW_i for input to the parameter correction logic, and 
   * dE/dz_i for further back-propogation  to earlier gates and/or cells.
   */

  int gate_no;
  int ns; // dimension of state signal + 1 (for bias)
  int nx; // dimension of input signal + 1
  
  Matrix<Matrix<double>>& W;
  /* Each W(i,j) (i =0,1,2, j = 0,1,2,3)) is an n_k x n_k+1 matrix of 
   * weights, where k = (s.s,x). The first column of each matrix is bias.  
   * This works because the input signals have a constant 1 in component 0. 
   * NOTE: W(1,0) is not used because gate 0 doesn't have a state input. 
   */
  Matrix<Matrix<double>>& dE_dW; // weight gradient
  RowVector<double>& dE_dv; // partials w.r.t readout
  RowVector<double>& dE_ds; // partials w.r.t state 
  Array<Matrix<double>> no_bias; // slice of each W(i,j) to remove bias

  // the following three vectors are saved for backprop
  ColVector<double> v; // readout (saved for backprop
  ColVector<double> s; // state
  ColVector<double> x; // cell input
  ColVector<double> g; // gate output
 public:
  Gate(Matrix<Matrix<double>>& W0, // Model weights & biases
       /* The next four items are stored in LSTM and referenced everywhere
        *else.  They are updated by each call to Gate::b_step.
        */
       Matrix<Matrix<double>>& dE_dW0,
       RowVector<double>& dE_dv0,
       RowVector<double>& dE_ds0,
       int gn) : W(W0), dE_dW(dE_dW0), dE_dv(dE_dv0), dE_ds(dE_ds0){
    gate_no = gn;
    n_s = W(0,0).ncols();
    n_x = W(0,2).ncols();
    v.reset(n_s+1);
    s.reset(n_s+1);
    x.reset(n_x+1);
    g.reset(n_s); // these are saved for backprop
    no_bias.reset(3);
    for(i = 0;i < 2;i++) no_bias[i] = W(i,gn).slice(0,1,n_s,n_s);
    no_bias[2] = W(2,gn).slice(0,1,n_s,n_x);
  }

  ColVector<double> f_step(Array<ColVector<double>>& z,ColVector<double> x);
  void b_step(RowVector<double>& dE_dg)
};

struct LSTMcell {
   /* The parameters are 11 matrices arranged in a 3x4 grid W.  W(i,j) 
   * (i = 0,1,2; j = 0,1,2,3) is the 
   * matrix which multiplies input vector i(0=v,1=s,2=x) in gate j 
   * (j = 1,2,3).  W(1,0) is a dummy matrix because gate 0 has no state
   * input.
   *
   * n_s+1 (resp. n_x+1) is the dimension of the state (s) and readout (v) vectors (resp. input vector x).  Note that the actual number of state 
   * (resp. input) variables is n_s (resp. n_x) because component 0
   * of each type is has constant value 1 (or 0) to provide for bias.
   *
   * W(i,2) has dimension n_s by n_x+1 (0<=i<=3).  All other parameter 
   * matrices have dimension n_s by n_s+1.  
   *
   * The readout signal is basically just an externalized state signal.
   * It typically might be converted to a probability (or weight)vector 
   * if the LSTM is coupled to some standard output distro (e.g. gaussian)
   *
   * The LSTM itself isa linear array of cells, wired up in series.
   * Each call takes a separate input vector x and the readout and state
   * vectors from the previous call as the input to gates 0-3. 
   * There are two outputs from the current cell which are also the
   * inputs to the next cell:
   * 1. current state (s), a sum of the gated previous state output and a 
   * gated non-linear combination of input and last cell output
   * 2. readout (v), a squashed and gated version of the current state
   */
  friend class LSTM;
  friend class Gate;
  int n_s; // no. of  state and output variables
  int n_x; // no. of input variables
  Array<Gate> gate; // four copies of struct Gate
  Matrix<Matrix<double>>& W;
  Matrix<Matrix<double>>& dE_dW;
  Array<RowVector<double>>& dE_dv;
  Array<RowVector<double>>& dE_ds;
  
  ColVector<double> r; // intermediate output

  LSTMcell(Matrix<Matrix<double>>& W0,
           Matrix<Matrix<double>>& dE_dW0,
           Array<RowVector<double>>& dE_dv0,
           Array<RowVector<double>>& dE_ds0) :
    W(W0), dE_dW(dE_dW0), dE_dv(dE_dv0), dE_ds(dE_ds0), gate(4)
  {
    assert(W0.nrows() == 3 && W0.ncols() == 4);
    for(int j = 0;j < 4;j++){
      gate[j] = Gate(W0,dE_dW0,dE,dv0,dE,ds0,j);
    }
    r.reset(n_s+1);
  }
  void forward_step(Array<ColVector<double>>& z, ColVector<double>& x);
  void backward_step(Array<RowVector<double>>& dE_dz, RowVector<double> dE_n);
  /* forward input/output is z = (v,s) and (input only) x
   * backward input/output is dE/dz = (dE/dv,dE/ds)
   * in addition, the matrix dE/dW is updated on each call
   */
};

struct LSTM {
  Array<LSTMcell> cells;
  int ncells;
  //  int n_s;
  //  int n_x;
  Matrix<double> data; 
  Matrix<double> output;
  Matrix<Matrix<double>> parameters; //3x4 grid of weight/bias matrices
public:
  LSTM(Matrix<double>& d, Matrix<double>& o, Matrix<Matrix<double>>& p);
  void train(int max_iters, double pct=1.0, double learn = .1, double eps = 1.0e-8);
};

#endif
