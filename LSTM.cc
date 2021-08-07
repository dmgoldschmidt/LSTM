#include <iostream>
#include <cassert>
#include "util.h"
#include "LSTM.h"
using namespace std;

void work_around(Matrix<double>& A, Matrix<double>&B, Matrix<double>& C){// A += B*C
  for(int i = 0;i < A.nrows();i++){
    for(int j = 0;j < A.ncols();j++){
      for(int k = 0;k < B.ncols();k++) A(i,j) += B(i,k)*C(k,j);
    }
  }
}

ColVector<double> augment(ColVector<double>& x){
   ColVector<double> y(x.nrows() + 1);
   y[0] = 1.0;
   for(int i = 0;i < x.nrows();i++) y[i+1] = x[i];
   return y;
 }

//double sigma(double u){return 1/(1+exp(-u));}

ColVector<double>& squash(ColVector<double>& x){
  for(int i = 0;i < x.nrows();i++) x[i] = sigma(x[i]);
  return x;
}

ColVector<double>& bulge(ColVector<double>& x, int n){ // n = 0 if unaugmented, 1 if augmented
  for(int i = n;i < x.nrows();i++) x[i] =2*sigma(x[i])-1;
  return x;
}

std:: ostream& operator <<(std::ostream& os, Cell& c){
  os << "dim(state): "<<c.n_s<<", dim(input): "<<c.n_x<<endl;
  os << "readout: "<<c.v.Tr()<<"state: "<<c.s.Tr()<<endl;
  os << "dE_dv: "<<c.dE_dv<<" dE_ds: "<<c.dE_ds<<endl;
  os << "W:\n"<<c.W;
  os << "dE_dW:\n"<<c.dE_dW;
  return os;
}


ColVector<double> Gate::f_step(ColVector<double>& v0, //readout
                                     ColVector<double>& s0, // state
                                     ColVector<double>& x0) // input
{
  int j = gate_no;
  v = v0.copy(); // save inputs for b_step (backprop)
  s = s0.copy(); // ditto
  x = x0.copy();
  g = W(0,j)*v + W(2,j)*x; 
  if(j > 0){
    g += W(1,j)*s; // apply weights to the input
    g  = squash(g);  // and save it for backward_step
  }
  else g = augment(bulge(g)); // gate 0 only
  return g;
}

void Gate::b_step(RowVector<double>& dE_dg)
/* Update partials w.r.t model weights, readout and state signals
 * NOTE: dE/d(anything) is an unaugmented row vector! 
 * dE_dW(i,j) needs to be transposed before the final gradient is 
 * computed. 
 */

{

  int j = gate_no;
  double u = (j == 0 ? .5 : 1);
  for(int i = 0;i < n_s;i++)dE_dg[i] = dE_dg[i]*g[i]*(1-g[i])*u;
  dE_dv += dE_dg*no_bias[0];
  if(gate_no > 0) dE_ds += dE_dg*no_bias[1];
  /* dE_dv and dE_ds are back_propagated to the previous gate or cell. 
   * Current values are read by Cell at input.
   * Next, we update the partials wrt model parameters
   */
  // update weight gradients (v,s,x were saved by f_step)
  work_around(dE_dW(0,j),v,dE_dg);// dE_dW(0,j)+=v*dE_dg; 
  if(j > 0) work_around(dE_dW(1,j),s,dE_dg);//dE_dW(1,j) += s*dE_dg;
  work_around(dE_dW(2,j),x,dE_dg);//  dE_dW(2,j) += x*dE_dg;
}
 
/*********************** begin Cell code here */

inline ColVector<double> throttle(ColVector<double>& x, ColVector<double>& g){
  // x may be augmented or not
  ColVector<double> y = x.copy(); // output has dim(g) (typically n_s)
  int n = (x.nrows() == g.nrows() ? 0:1); 
  for(int i = 0;i < g.nrows();i++) y[i+n] = x[i+n]*g[i];
  return y;
}

inline ColVector<double> throttle(RowVector<double>& x, ColVector<double>& g){ // this version is for backprop
  // x may be augmented or not
  RowVector<double> y = x.copy(); // output has dim(x) (typically n_s)
  int n = (g.nrows() == y.ncols() ? 0: 1);
  for(int i = 0;i < y.ncols();i++) y[i] = x[i]*g[n+i];
  return y;
}

void Cell::forward_step(ColVector<double>& x){ // v and s are Cell variables
  gate[0].f_step(v,s,x);
  gate[1].f_step(v,s,x);
  gate[2].f_step(v,s,x);
  //  ColVector<double> g1 = augment(gate[0].g);
  s = throttle(s, gate[2].g) + throttle(gate[0].g, gate[1].g);
  gate[3].f_step(v,s,x);
  r = bulge(s,1); // argument_2 = 1 for augmented input
  v = throttle(r, gate[3].g);
}



void Cell::backward_step(RowVector<double>& dE_dxn){
  dE_dv += dE_dxn; // combine local gradient
  dE_dr = throttle(dE_dv,gate[3].g);
  for(int i = 0;i < n_s;i++)dE_ds[i] += dE_dr[i]*(1-r[i]*r[i])/2;
  dE_dg = throttle(dE_dv,r);
  gate[3].b_step(dE_dg); // r was saved during the forward step
  
  // OK, now dE_ds, dE_dv, and dE_dW(i,3) are updated past gate 3.
  // save for gate[1] backprop before pushing through the next operation
  RowVector<double> dE_ds1 = throttle(dE_ds,gate[2].g); 
  dE_dg = throttle(dE_ds,gate[2].s);
  gate[2].b_step(dE_dg);
  dE_ds += dE_ds1;
  // OK, now we're past gate[2]
  dE_dg = throttle(dE_ds,gate[0].g);
  dE_ds1 = throttle(dE_ds,gate[1].g);
  gate[1].b_step(dE_dg);
  gate[0].b_step(dE_ds1);
}
  
// LSTM::LSTM(Matrix<double>& d, Matrix<double>& o, Matrix<Matrix<double>>& p) :
//   data(d),output(o),parameters(p) {
  
//   Cell::static_initializer(output.nrows(),data.nrows());
//   int& n_s = Cell::n_s;
//   for(int i = 0;i < 4;i++){
//     assert(data.nrows() == parameters(i,2).ncols() && data.nrows() == Cell::n_x); // input dimension
//     for(int j = 0;j < 2;j++)assert(output.nrows() == parameters(i,j).ncols() && output.nrows() == n_s);
//   }
//   ncells = data.ncols(); 
//   cells.reset(ncells+2); // first and last cell are initializers

//   cells[0].reset(nullptr,&cells[1],parameters);
//   cells[ncells+1].reset(&cells[ncells],nullptr,parameters);
//   for(int i = 1;i <= ncells;i++)cells[i].reset(&cells[i-1],&cells[i+1],parameters);
//   // initialize first and last cells
//   cells[0].w[0].copy(Cell::zero);
//   cells[0].w[1].copy(Cell::zero);
//   cells[ncells+1].d_ds.copy(Cell::zero);
//   cells[ncells+1].d_dv.copy(Cell::zero);
// }

// void LSTM::train(int niters,double pct,double learn,double eps){

