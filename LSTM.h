#ifndef LSTM_H
#define LSTM_H
#include <iostream>
#include <cassert>
#include <cmath>
#include <fenv.h>
#include "using.h"
#include "util.h"
#include "GetOpt.h"
#include "Matrix.h"
#include "Array.h"

ColVector<double> augment(ColVector<double>& x);

inline double sigma(double u){return 1/(1+exp(-u));}

void squash(ColVector<double>& x, double b = .9);

void bulge(ColVector<double>& x,int n = 0);

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

  int gn;
  int n_s; // no. of state parameters
  int n_x; // no. of input parameters
  
  Matrix<double> W_v; // Each gate has a shallow copy of the relevant weight/
  Matrix<double> W_s; // bias matrices
  Matrix<double> W_x;
  /* Each W_k is an n_s x n_k+1 matrix of weights and biases, where 
   * k = (s,s,x). The first column of each matrix is bias.  
   * This works because the input signals have a constant 1 in component 0. 
   * NOTE: gate 0 doesn't use W_s because it has no state input. 
   */
  Matrix<double> dE_dW_v, dE_dW_s, dE_dW_x; // weight/bias gradients (slices of dE_dW)
  RowVector<double> dE_dv; // partials w.r.t readout for backprop
  RowVector<double> dE_ds; // partials w.r.t state for backprop
  Matrix<double> U_v, U_s, U_x; // slice of each W_k to remove bias

  // v_old, s_old, and x are saved for backprop
  ColVector<double> v,v_old; // readout (fed forward and modified)
  ColVector<double> s,s_old; // state   (ditto)
  ColVector<double> x; // cell input from user (unique to each cell) 
  
  ColVector<double> g; // gate output (saved for backprop
 public:
  Gate(void){}
  void reset(Matrix<double>& W, // All model weights & biases.
                                // Slices are computed below.
             Matrix<double>& dE_dW, // All model gradients  -- shallow copied
                                    // and updated by b_step
             RowVector<double>& dE_dv0, //backprop variables --shallow copied                                        //updated, and saved by b_step)
             RowVector<double>& dE_ds0,
             int n_s0, int n_x0, int gn0) {
    n_s = n_s0; n_x = n_x0; gn = gn0;
    dE_dv = dE_dv0;
    dE_ds = dE_ds0;

    W_v = W.slice(0,gn*(n_s+1),n_s,n_s+1);
    W_s = W.slice(n_s,gn*(n_s+1),n_s,n_s+1);
    W_x = W.slice(2*n_s,gn*(n_x+1),n_s,n_x+1);

    dE_dW_v = dE_dW.slice(0,gn*(n_s+1),n_s,n_s+1);
    dE_dW_s = dE_dW.slice(n_s,gn*(n_s+1),n_s,n_s+1);
    dE_dW_x = dE_dW.slice(2*n_s,gn*(n_x+1),n_s,n_x+1);

    U_v = W_v.slice(0,1,n_s,n_s); // we're slicing off the bias column here
    U_s = W_s.slice(0,1,n_s,n_s);
    U_x = W_x.slice(0,1,n_s,n_x);

    // v_old,s_old,x,g are all saved for backprop
    g.reset(n_s); 
  }
  
  // Gate& operator=(const Gate& gg){
  //   gate_no = gg.gate_no; n_s = gg.n_s; n_x = gg.n_x; W = gg.W; dE_dW = gg.dE_dW;
  //   dE_dv = gg.dE_dv; dE_ds = gg.dE_ds; v = gg.v; s = gg.s;x = gg.x; g = gg.g; no_bias = gg.no_bias;
  //   return *this;
  // }
  
  ColVector<double> f_step(ColVector<double>& v, ColVector<double>& s,
                           ColVector<double>& x);
  void b_step(RowVector<double>& dE_dg);
};

struct Cell {
  /* A Cell consists of 4 Gates which basically apply affine maps (followed by a non-linear "squash")
   * in series to the 3 signals v,s,x.  They are all "augmented" column vectors, meaning component 0
   * is a constant 1.0 to account for bias (the affine additive).
   *
   * The affine parameters are packed into a 3n_s x 4n Matrix<double> W, where n = max(n_s, n_x)+1 and
   * n_s = no. of state & readout variables, n_x =no. of input variables.
   * Each gate instance unpacks the appropriate slices of W into a set of 3 weight/bias matrices W_i 
   * which multiply augmented vectors v, s, or x thereby applying an affine map.  
   * gate 0 has no state input so there are actually only 11 affine matrices.
   *
   * The readout signal v is basically just an externalized state signal.
   * It typically might be converted to a probability (or weight)vector 
   * if the LSTM is coupled to some standard output distro (e.g. gaussian)
   *
   * The LSTM user converts v to an estimated target signal, and in training mode computes an error
   * gradient from the ground truth which is fed back to the cell.
   * 
   * The LSTM itself is a linear array of identical Cells, wired up in series.  The Cells are fed input
   * from consecutive terms of a time series.
   * Each Cell takes a separate input vector x and the readout and state
   * vectors from the previous call as the input to gates 0-3. 
   * There are two outputs from the current Cell which are also the
   * inputs to the next Cell:
   * 1. current state (s)
   * 2. readout (v), a squashed and gated version of the current state
   */
  friend class LSTM;
  friend class Gate;
  int n_s; // no. of  state and output variables
  int n_x; // no. of input variables
  Array<Gate> gate; // each Cell has a separate set of four Gates.
  // The following six items are defined in LSTM and updated by every Cell/Gate
  ColVector<double> v;
  ColVector<double> s;
  Matrix<double> W;
  Matrix<double> dE_dW;
  RowVector<double> dE_dv;
  RowVector<double> dE_ds;
  
  ColVector<double> r; // intermediate output defined here to avoid dynamic memory thrashing
  RowVector<double> dE_dr;
  RowVector<double> dE_dg;
  Cell(void){}
  void reset(
       int n_s0, // no. of state parameters
       int n_x0, // no. of input parameters
       ColVector<double>& v0, 
       ColVector<double>& s0,
       Matrix<double>& W0,
       Matrix<double>& dE_dW0,
       RowVector<double>& dE_dv0,
       RowVector<double>& dE_ds0);
  void forward_step(ColVector<double>& x);
  void backward_step(RowVector<double>& dEn_dv);
  /* forward input is v,s and x, output is v,s.
   * backward input is dE_n/dv, dE/dv,dE/ds
   * in addition, the matrix dE/dW is updated at each backward_step
   */
};

std:: ostream& operator <<(std::ostream& os, Cell& c);

struct LSTM {
  Array<Cell> cell;
  int ncells;
  int n_s;
  int n_x;
  Array<ColVector<double>> data; 
  Matrix<double> output;
  Matrix<double> W;
  Matrix<double> dE_dW;
  ColVector<double> v;
  ColVector<double> s;
  RowVector<double> dE_ds;
  RowVector<double> dE_dv;
public:
  LSTM(int ns0, int nx0, int nc, Array<ColVector<double>>& d,
       Matrix<double>& o,Matrix<double>& w);
  void train(int niters,double a = .001,double eps = 1.0e-8, 
             double b1 = .9,double b2 = .999);
};

#endif
