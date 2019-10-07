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

double sigma(double x){return 1/(1+exp(-x));}

void G_c(ColVector<double>& x, ColVector<double>& g){
  int dim = x.nrows();
  for(int i = 0; i < dim;i++) g[i] = sigma(x[i]);
}
void dG_c(ColVector<double>& x, ColVector<double>& g){
  int dim = x.nrows();
  for(int i = 0; i < dim;i++)g[i] = sigma(x[i])*sigma(-x[i]);
}
void G_d(ColVector<double>& x, ColVector<double>& g){
  int dim = x.nrows();
  for(int i = 0;i < dim;i++) g[i] = 2*sigma(x[i])-1;
}
void dG_d(ColVector<double>& x, ColVector<double>& g){
  int dim = x.nrows();
  for(int i = 0;i < dim;i++)g[i] = 2*sigma(x[i])*sigma(-x[i]);
}

class LSTMcell {
  /* There are three gates:
   * 1. state update: controls a squashed linear combination of the input and the last cell readout
   *    gate parameters are a squashed linear combination of previous state, input, and last cell readout
   * 2. state pass-thru: controls the previous state input to the current state accumulator
   *    gate parameters are a squashed linear combination of previous state, input, and last cell readout
   * 3. output: controls the cell output which is just the squashed and gated current state
   *    gate parameters are a squashed linear combination of current state, input and last cell readout

   * There are two outputs:
   * 1. current state, an ungated sum of the gated previous state and the gated non-linear combination of 
        input and last cell output
   * 2. readout, a squashed and gated version of the current state

   */
  friend class LSTM;
  static int s; // dimension of the state vector
  static int d; // dimension of the input vector

  /* backward pass temporaries
   * they are esssentially scratch computations that don't need to be saved
   * so they are static to minimize dynamic memory thrashing. Each instance
   * overwrites whatever is there from the last time step.
   *
   */
  static ColVector<double> alpha_du; 
  static ColVector<double> alpha_cu;
  static ColVector<double> alpha_cs;
  static ColVector<double> alpha_cr;

  static ColVector<double> d_cu; // G_c'(a_cu)
  static ColVector<double> d_cs; // G_c'(a_cs)
  static ColVector<double> d_cr; // G_c'(a_cr)
  static ColVector<double> d_s; // G_d'(state)
  static ColVector<double> d_du; // G_d'(a_du)

  static ColVector<double> chi; // dE/dv
  static ColVector<double> rho; // dE/dr
  static ColVector<double> gamma; // dE/dg
  static ColVector<double> psi; // dE/ds

  /* these are the actual gradients we will use to update parameters for
   * the next iteration.  Each instance adds its contribution.
   */
  static Matrix<double> dE_dW_xdu;
  static Matrix<double> dE_dW_vdu;
  static Matrix<double> dE_dW_vcu;
  static Matrix<double> dE_dW_xcu;
  static Matrix<double> dE_dW_scu;
  static Matrix<double> dE_dW_vcs;
  static Matrix<double> dE_dW_xcs;
  static Matrix<double> dE_dW_scs;
  static Matrix<double> dE_dW_xcr;
  static Matrix<double> dE_dW_scr;
  static Matrix<double> dE_dW_vcr;

  // references to the model parameters
  Matrix<double>& W_xdu; // from input to a_du 
  Matrix<double>& W_vdu; // from lastcell a_du

  Matrix<double>& W_vcu; // from lastcell to a_cu 
  Matrix<double>& W_xcu; // from input a_cu
  Matrix<double>& W_scu; // from state to a_cu

  Matrix<double>& W_vcs; // from lastcell to a_cs 
  Matrix<double>& W_xcs; // from input to a_cs
  Matrix<double>& W_scs; // from state to a_cs

  Matrix<double>& W_xcr; // from input to a_cr 
  Matrix<double>& W_scr; // from state to a_cr
  Matrix<double>& W_vcr; // from lastcell to a_cr

  /* these parameters are instance specific but are defined here
   * so that we don't thrash dynamic memory by re-allocating each
   * iteration.   
   * they are forward pass temporaries which are used in the backward pass
   * which is why they are not static
   */
  ColVector<double> u; // G_d(input sum)
  ColVector<double> r; // G_d(state);
  ColVector<double> a_cu; // (input-gate sum)
  ColVector<double> a_cs; // (state-gate sum)
  ColVector<double> a_cr; // (readout-gate sum)
  ColVector<double> a_du; // (input sum)
  ColVector<double> g_cu; // G_c(a_cu)
  ColVector<double> g_cs; // G_c(a_cs)
  ColVector<double> g_cr; // G_c(a_cr)
  // outputs
  ColVector<double> f_chi; // these two are passed to time t-1 in the backward pass (ala BPTT)
  ColVector<double> f_psi;
  
  ColVector<double> _state;
  ColVector<double> _readout;
  LSTMcell* prev_cell;
  LSTMcell* next_cell;

  // input
  ColVector<double> _input;

public:
  LSTMcell(void);
  void reset(LSTMcell* pc, LSTMcell* nc, Array<Matrix<double>>& p);
  static void static_initialization(int ss, int dd);
  ColVector<double> state(void){return _state;}
  ColVector<double> readout(void){return _readout;}
  void forward_step(ColVector<double>& x);
  void backward_step(ColVector<double>& dE_dv);
};

class LSTM {
  Array<LSTMcell> cells;
  int ncells;
  int nstates;
  Matrix<double> data;
  Matrix<double> output;
  Array<Matrix<double>> parameters;
public:
  LSTM(int ns, Matrix<double>& d, Matrix<double>& o, Array<Matrix<double>>& p) :
    nstates(ns), data(d),output(o),parameters(p){
    
    ncells = d.ncols(); 
    cells.reset(ncells+2); // first and last cell are initializers
    cells[0]._state = cells[0]._readout = 0;
    cells[ncells+1].f_chi = cells[ncells+1].f_psi = 0;
    LSTMcell::static_initializer(
    for(int i = 1;i <= ncells;i++) cells[i].
    
    


#endif
