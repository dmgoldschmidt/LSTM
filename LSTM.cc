#include <iostream>
#include <cassert>
#include "util.h"
#include "LSTM.h"
using namespace std;

ColVector<double> Gate::f_step(ColVector<double>& v0, //readout
                                     ColVector<double>& s0, // state
                                     ColVector<double>& x0) // input
{
  int j = gate_no;
  v = v0.copy(); // save inputs for b_step
  s = s0.copy(); // ditto
  x = x0.copy();
  g = W(0,j)*v + W(2,j)*x; 
  if(j > 0){
    g += W(1,j)*s; // apply weights to the input
    return g  = squash(g);  // and save it for backward_step
  }
  else return g = augment(bulge(g)); // gate 0 only

}

void Gate::b_step(RowVector<double>& dE_dg)
/* Update partials w.r.t model weights, readout and state signals
 * NOTE: dE/d(anything) is an unaugmented row vector! 
 * dE_dW(i,j) needs to be transposed before the final gradient is 
 * computed. 
 */

{

  int j = gate_no;

  for(int i = 0;i < n_s;i++)dE_dg[i] = dE_dg[i]*g[i]*(1-g[i]);
  dE_dv += dE_dg*no_bias[0];
  if(gate_no > 0) dE_ds += dE_dg*no_bias[1];
  /* dE_dv and dE_ds are back_propagated to the previous gate or cell. 
   * Current values are read by LSTMcell at input.
   * Next, we update the partials wrt model parameters
   */
  // update weight gradients (v,s,x were saved by f_step)
  dE_dW(0,j) += v*dE_dg; 
  if(j > 0)dE_dW(1,j) += s*dE_dg;
  dE_dW(2,j) += x*dE_dg;
}

/*********************** begin LSTMcell here */

inline void throttle(ColVector<double>& x, ColVector<double>& g){
  // x may be augmented or not
  ColVector<double> y = x.copy();
  int n = (x.nrows() == g.nrows() ? 0:1); 
  for(int i = 0;i < g.nrows();i++) y[i+n] = x[i+n]*g[i];
  return y;
}

void LSTMcell::forward_step(Array<ColVector<double>>& z,
                             ColVector<double>& x){
  ColVector<double>& v = z[0];
  ColVector<double>& s = z[1];
   
  s = throttle(s,gate[2].f_step(z,x)) +
    augment(throttle(gate[0].f_step(z,x),gate[1].f_step(z,x)));
  r = augment(bulge(s));
  v = throttle(r,gate[3].f_step(z,x));
}



void LSTMcell::backward_step(RowVector<double>& dE_n){
  dE_dv += dE_n; // combine local gradient
  dE_dr = throttle(dE_dv,cell[3].g);
  for(i = 0;i < n_s;i++)dE_ds[i] += dE_dr[i]*(1-r[i]*r[i])/2;
  gate[3].b_step(throttle(dE_dv,r)); // r was saved during the forward step

  // OK, now dE_ds, dE_dv, and dE_dW(i,3) are updated past gate 3.
  dE_ds1 = dE_ds.copy();
  // save for gate[1] backprop before pushing through the next operation
  dE_ds = throttle(dE_ds,gate[2].g); 
  gate[2].b_step(throttle(dE_ds,gate[2].s));
  
  // OK, now we're past gate[2]
  gate[1].b_step(throttle(dE_ds1,gate[0].g));
  gate[0].b_step(throttle(dE_ds1,gate[1].g));
}
  
LSTM::LSTM(Matrix<double>& d, Matrix<double>& o, Matrix<Matrix<double>>& p) :
  data(d),output(o),parameters(p) {
  
  LSTMcell::static_initializer(output.nrows(),data.nrows());
  int& n_s = LSTMcell::n_s;
  for(int i = 0;i < 4;i++){
    assert(data.nrows() == parameters(i,2).ncols() && data.nrows() == LSTMcell::n_x); // input dimension
    for(int j = 0;j < 2;j++)assert(output.nrows() == parameters(i,j).ncols() && output.nrows() == n_s);
  }
  ncells = data.ncols(); 
  cells.reset(ncells+2); // first and last cell are initializers

  cells[0].reset(nullptr,&cells[1],parameters);
  cells[ncells+1].reset(&cells[ncells],nullptr,parameters);
  for(int i = 1;i <= ncells;i++)cells[i].reset(&cells[i-1],&cells[i+1],parameters);
  // initialize first and last cells
  cells[0].w[0].copy(LSTMcell::zero);
  cells[0].w[1].copy(LSTMcell::zero);
  cells[ncells+1].d_ds.copy(LSTMcell::zero);
  cells[ncells+1].d_dv.copy(LSTMcell::zero);
}

void LSTM::train(int niters,double pct,double learn,double eps){
}
