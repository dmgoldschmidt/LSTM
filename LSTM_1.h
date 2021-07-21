#ifndef LSTM1_H
#define LSTM1_H
#include <iostream>
#include <cassert>
#include <cmath>
#include <fenv.h>
#include "util.h"
#include "GetOpt.h"
#include "Matrix.h"
#include "Array.h"

using namespace std;

 

struct Gate {
  /* A gate is just the function g = sigma(W_v*v + W_s*s + W_x*x). The products are matrix*vector products.
   * It is called a gate because g is component-wise multiplied by an information signal z = (v,s,or x)
   * which reduces the amplitude of z.  So a component of g near 1 lets most of the corresponding
   * component of z pass through, while a value near zero blocks most of that component.
   
   * The gate also operates in back-propagation mode.  In this mode, the input is dE/dg where E is some
   * sort of error signal.  The gate then computes dg/dW_i and dg/dz_i which are then multiplied by dE/dg
   * to get dE/dW_i for input to the parameter correction logic, and dE/dz_i for further back-propogation
   * to earlier gates and cells.
   */

  RowVector<Matrix<double>> W;
  ColVector<double> g;
  RowVector<Matrix<double>> dg_dw;
  RowVector<Matrix<double>> dg_dW;
 public:
  Gate(void){}
  void reset(RowVector<Matrix<double>> W0);
  ColVector<double> operator()(RowVector<ColVector<double>>& z);
};

struct LSTM_1cell {
  /* A cell consists of four gates.  cell[0] is special since a) the input vector s is always zero,
   * and b) the output g is modified to 2*g-1. 
   * The parameters are 11 matrices arranged in a 4x3 grid W.  W(i,j) (i,j = 1,2,3) is the 
   * matrix which multiplies input vector j(0=v,1=s,2=x) to gate i.  W(0,1) is a dummy matrix.
   * n_s (resp. n_x) is the dimension of the state (s) and readout (v) vectors (resp. input vector x). 
   * W(i,2) has dimension n_x by n_s (0<=i<=3).  All other parameter matrices have dimension n_s by n_s.  
   * The LSTM itself is a linear array of cells. Each one takes a separate input vector x and the readout
   * vector from the previous cell as the input to gates 0-3. The state output from the previous cell is input
   * to gates 1 and 2.  All cells use and update the same 11 parameter matrices.
   * There are two outputs:
   * 1. current state, an ungated sum of the gated previous state and the gated non-linear combination of 
        input and last cell output
   * 2. readout, a squashed and gated version of the current state
   */
  friend class LSTM_1;
  friend class Gate;
  static int n_s; // dimension of the state and output vectors
  static int n_x; // dimension of the input vector
  static ColVector<double> zero;
  static RowVector<double> d_dg; // temporary storage during backprop
  Array<Gate> gate;
  // ColVector<double>& s; // state
  // ColVector<double>& v; // readout
  ColVector<double> r; // tanh(s/2)
  RowVector<ColVector<double>> w;
  LSTM_1cell* prev_cell;
  LSTM_1cell* next_cell;
  // ColVector<double>& input;
  
  // Backprop variables: input from next cell, output to prev cell
  RowVector<double> d_ds; // d(forward error)/d(state)
  RowVector<double> d_dv; // d(forward error)/d(readout) 


  LSTM_1cell(void) {}
  void reset(LSTM_1cell* pc, LSTM_1cell* nc, Matrix<Matrix<double>> W0);
  void forward_step(ColVector<double>& x);
  void backward_step(RowVector<double>& dE_dv); // input is d(this_cell error)/d(readout)
  const ColVector<double>& readout(void){return w[0];}

  static void static_initializer(int ss, int xx);
  static double sigma(double x);
  static ColVector<double> squash(ColVector<double> x);
  static ColVector<double> bulge(ColVector<double> x);
};

struct LSTM_1 {
  Array<LSTM_1cell> cells;
  int ncells;
  //  int n_s;
  //  int n_x;
  Matrix<double> data; // include an extra row set to 1.0 if you want a bias term
  Matrix<double> output;
  Matrix<Matrix<double>> parameters; //4x3 grid of parameter matrices
public:
  LSTM_1(Matrix<double>& d, Matrix<double>& o, Matrix<Matrix<double>>& p);
  void train(int max_iters, double pct=1.0, double learn = .1, double eps = 1.0e-8);
};

#endif
