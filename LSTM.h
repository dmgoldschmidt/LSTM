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

 
double sigma(double u){return 1/(1+exp(-u));}
ColVector<double> squash(ColVector<double>& x){
  for(int i = 0;i < x.nrows();i++)x[i] = sigma(x[i]);
  return x;
}

ColVector<double> bulge(ColVector<double> x){
  for(int i = 0;i < x.nrows();i++) x[i] =2*x[i]-1;
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
   * Note that dE/dx is not computed because there is no back-progation.
   */

  int gate_no;
  Array<ColVector<double>> b;
  Array<Matrix<double>> W;
  /* Each pair W[i],b[i] (i =0,1,2) is a set of weights.  There are four
   * separate such sets in an LSTM cell, but the first gate in a cell is 
   * special because it doesn't have a state input. The same weights are
   * repeated in each cell.
   */
  ColVector<double> g;
  RowVector<Matrix<double>> dg_dz;
  RowVector<Matrix<double>> dg_dW;
 public:
  Gate(void){}
  void reset(Array<Matrix<double>> W0, int gn);
  ColVector<double> operator()(RowVector<ColVector<double>>& z,
                               int gate_no);
};

struct LSTMcell {
   /* The parameters are 11 matrices arranged in a 3x4 grid W.  W(i,j) 
   * (i = 0,1,2; j = 0,1,2,3) is the 
   * matrix which multiplies input vector i(0=v,1=s,2=x) in gate j 
   * (j = 1,2,3).  W(1,0) is a dummy matrix because gate 0 has no state
   * input.
   *
   * n_s (resp. n_x) is the dimension of the state (s) and readout (v) vectors (resp. input vector x).  Note that the actual number of state 
   * (resp. input) variables is n_s-1 (resp. n_x-1) because component 0
   * of each type is has constant value 1 (or 0) to provide for bias or
   * no bias
   *.    
   * W(i,2) has dimension n_x by n_s (0<=i<=3).  All other parameter 
   * matrices have dimension n_s by n_s.  The first row of each parameter
   * matrix is (1,0,..0) to preserve the leading value 1 of each signal.
   * The readout signal is basically just an externalized state signal.
   * It typically might be converted to a probability (or weight)vector 
   * if the LSTM is coupled to some standard output distro (e.g. gaussian)
   *
   * The LSTM itself is conceptually a linear array of cells, but in
   * implementation, we iterate calls to the same code with input from
   * the previous call. Each call takes a 
   * separate input vector x and the readout and state
   * vectors from the previous call as the input to gates 0-3. 
   * There are two outputs from the current cell which are also the
   * inputs to the next cell.:
   * 1. current state (s), a sum of the gated previous state output and a 
   * gated non-linear combination of input and last cell output
   * 2. readout (v), a squashed and gated version of the current state
   */
  friend class LSTM;
  friend class Gate;
  int n_s; // dimension of the state and output vectors
  int n_x; // dimension of the input vector
  //  static ColVector<double> zero;
  //  static RowVector<double> d_dg; // temporary storage during backprop
  Array<Gate> gate; // four copies of struct Gate
  
  // Backprop variables: input from next cell, output to prev cell
  RowVector<double> d_ds; // d(forward error)/d(state)
  RowVector<double> d_dv; // d(forward error)/d(readout) 


  LSTMcell(void) {}
  void reset(Matrix<Matrix<double>>& W0); // weights for all four gates
  void forward_step(Array<ColVector<double>>& z, ColVector<double>& x);
  void backward_step(Array<RowVector<double>>& dE_dz);
  /* forward input/output is z = (v,s) and (input only) x
   * backward input/output is dE/dv,dE/ds
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
  Matrix<Matrix<double> > parameters; //4x3 grid of parameter matrices
public:
  LSTM(Matrix<double>& d, Matrix<double>& o, Matrix<Matrix<double>>& p);
  void train(int max_iters, double pct=1.0, double learn = .1, double eps = 1.0e-8);
};

#endif
