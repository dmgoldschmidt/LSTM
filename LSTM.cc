#include <iostream>
#include <cassert>
#include "using.h"
#include "util.h"
#include "LSTM.h"
//using namespace std;

void work_around(Matrix<double> A, Matrix<double> B, Matrix<double> C){// A += B*C
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

void squash(ColVector<double>& x){
  for(int i = 0;i < x.nrows();i++) x[i] = sigma(x[i]);
}

void bulge(ColVector<double>& x, int n){ // n = 0 if unaugmented, 1 if augmented
  for(int i = n;i < x.nrows();i++) x[i] =2*sigma(x[i])-1;
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
  v = v0; 
  s = s0;
  // deep copy inputs for b_step (backprop)
  v_old = v0.copy(); 
  s_old = s0.copy(); 
  x = x0.copy();
  g = W_v*v + W_x*x; //apply affine maps to w,x  
  if(gn > 0){ // and s except at gate 0
    g += W_s*s; // apply weights to the input
    squash(g);  // and save it for backward_step
  }
  else{
    bulge(g); // gate 0 only
    g = augment(g);
  }
  cout <<format("gate[%d] at exit: g = ",gn)<<g.Tr();
  return g;
}

void Gate::b_step(RowVector<double>& dE_dg)
/* Update partials w.r.t model weights, readout and state signals
 * NOTE: dE/d(anything) is an unaugmented row vector! 
 * dE_dW_k needs to be transposed before the final gradient is 
 * computed. 
 */

{ 
  double u = (gn == 0 ? .5 : 1);
  cout << format("dE_dg pre-squash: %.9f",dE_dg[0])<<", post-squash:";
  for(int i = 0;i < n_s;i++)dE_dg[i] = dE_dg[i]*g[i]*(1-g[i])*u; // back up thru squash
  cout << dE_dg.Tr();
  dE_dv += dE_dg*U_v; // back up thru W_v (weights only!  bias has no effect here
  if(gn > 0) dE_ds += dE_dg*U_s; // and W_s
  /* dE_dv and dE_ds are back_propagated to the previous gate or cell. 
   * Current values are read by Cell at input.
   * Next, we update the partials wrt model parameters
   */
  // update weight/bias gradients (v_old,s_old,x were saved by f_step)
  cout << "about to update dE_dW_v.  v = "<<v.Tr()<<", dE_dW_v = "<<dE_dW_v;
  dE_dW_v += (v_old*dE_dg).Tr(); 
  if(gn > 0) dE_dW_s += (s_old*dE_dg).Tr();
  dE_dW_x += (x*dE_dg).Tr();
  cout << "gate "<<gn<<" b_step: dE_dg = "<<dE_dg<<", dE_dv = "<<dE_dv<<" , dE_ds = "<<dE_ds<<endl;
  cout << "dE_dW_v:\n"<<dE_dW_v<<"dE_dW_s:\n"<<dE_dW_s<<"dE_dW_x:\n"<<dE_dW_x;
  
}
 
/*********************** begin Cell code here */

inline ColVector<double> throttle(ColVector<double>& x, ColVector<double>& g){
  // x may be augmented or not
  ColVector<double> y = x.copy(); 
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

void Cell::reset(int n_s0, int n_x0, ColVector<double>& v0,
                 ColVector<double>& s0, Matrix<double>& W0,
                 Matrix<double>& dE_dW0, RowVector<double>& dE_dv0,
                 RowVector<double>& dE_ds0){
  n_s = n_s0; n_x = n_x0; v = v0; s = s0; W = W0; dE_dW = dE_dW0;
  dE_dv = dE_dv0; dE_ds = dE_ds0;
  assert(W0.nrows() >= 3*n_s && W0.ncols() >= 4*(std::max(n_s,n_x)+1));
   gate.reset(4);
  for(int j = 0;j < 4;j++){
    gate[j].reset(W0,dE_dW0,dE_dv0,dE_ds0,n_s,n_x,j);
  }
  r.reset(n_s+1);
  dE_dr.reset(n_s);
  dE_dg.reset(n_s);
}

void Cell::forward_step(ColVector<double>& x){ // v and s are Cell variables
  gate[0].f_step(v,s,x);
  gate[1].f_step(v,s,x);
  gate[2].f_step(v,s,x);
  //  ColVector<double> g1 = augment(gate[0].g);
  s.copy(throttle(s, gate[2].g) + throttle(gate[0].g, gate[1].g));
  s[0] = 1.0;
  // s <- s .* gate[2].g + (bulge(gate[0].g) .* gate[1].g)
  gate[3].f_step(v,s,x);
  r = s.copy();
  bulge(r,1); 
  v.copy(throttle(r, gate[3].g)); // v <- bulge(s) .* gate[3].g
}

void Cell::backward_step(RowVector<double>& dEn_dv){
  cout << "Backward step.  dEn_dv:\n"<<dEn_dv;
  dE_dv += dEn_dv; // combine local error gradient
  dE_dr = throttle(dE_dv,gate[3].g);
  for(int i = 0;i < n_s;i++)dE_ds[i] += dE_dr[i]*(1-r[i]*r[i])/2; // update dE_ds from local error gradient
  dE_dg = throttle(dE_dv,r);
  gate[3].b_step(dE_dg); // r was saved during the forward step
  
  // OK, now dE_ds, dE_dv, and dE_dW(i,3) are updated past gate 3.
  // save for gate[1] backprop before pushing through the next operation
  RowVector<double> dE_ds1 = throttle(dE_ds,gate[2].g); 
  cout << format("gate[2] pre_throttle dE_ds: %f, gate[2].s: %f\n",dE_ds[0],gate[2].s);
  dE_dg = throttle(dE_ds,gate[2].s);
  gate[2].b_step(dE_dg);
  dE_ds += dE_ds1;
  // OK, now we're past gate[2]
  dE_dg = throttle(dE_ds,gate[0].g);
  dE_ds1 = throttle(dE_ds,gate[1].g);
  gate[1].b_step(dE_dg);
  gate[0].b_step(dE_ds1);
}
  
LSTM::LSTM(int ns0, int nx0, int nc, Matrix<double>& d, Matrix<double>& o,
           Matrix<double>& w) : n_s(ns0),n_x(nx0),ncells(nc),data(d),
                               output(o),W(w){
  int n = std::max(n_s,n_x);
  dE_dW.reset(3*n_s,4*(n+1));
  v.reset(n_s+1);
  s.reset(n_s+1);
  dE_ds.reset(n_s);
  dE_dv.reset(n_s);

  assert(data.ncols() == n_x+n_s); // contexts + goals
  assert(ncells <= data.nrows());
  if(ncells == 0) ncells = data.nrows();
  cell.reset(ncells); 

  for(int i = 0;i < ncells;i++)
    cell[i].reset(n_s,n_x,v,s,W,dE_dW,dE_dv,dE_ds);
  cout << "sizeof(Cell) = "<<sizeof(cell[0])<<endl;
}
 
void LSTM::train(int niters,double eps, double a,double b1,
                 double b2){

  ColVector<double> x(n_x+1);
  x[0] = 1;
  Array<double> E(ncells); // readout error at cell i
  Array<RowVector<double>> dEi_dv(ncells);
               // gradient of readout error at cell i
  for(int i = 0;i < ncells;i++) dEi_dv[i].reset(n_s);
  double E_tot = 1.0;
  double b1t = b1;
  double b2t = b2;
  Matrix<double> M(W), V(W);
  M.fill(0); V.fill(0);;
  int it = 0;
  
  while( it++ < niters && E_tot > eps){
    E_tot = 0;
    v.fill(0); v[0] = 1.0;
    s.fill(0); s[0] = 1.0;
    dE_ds.fill(0);
    dE_dv.fill(0);
    dE_dW.fill(0);
    
    for(int t = 0;t < data.nrows()-ncells;t += ncells){
      for(int i = 0;i < ncells;i++){
        for(int j = 0;j < n_x;j++) x[j+1] = data(i+t,j);
        cout << format("\nCell[%d] before forward step\n",i)<<cell[i];
        cell[i].forward_step(x);
        E[i] = 0;
        for(int j = 0;j < n_s;j++){
          double delta = cell[i].v[j+1] - data(i,n_x+j);
          E[i] += delta*delta; 
          dEi_dv[i][j] = 2*delta;
          output(i,j) = cell[i].v[j+1];
        }
        E_tot += E[i];
        cout << format("\nCell[%d] after forward_step:\n",i)<<cell[i];
      }

      for(int i = ncells-1;i >= 0;i--){
        cout << format("\nCell[%d] before backward_step:\n",i)<<cell[i];
        cell[i].backward_step(dEi_dv[i]);
        cout << format("\nCell[%d] after backward_step:\n",i)<<cell[i];
      }
    }
    E_tot /= data.nrows();
  
    
    // OK, data pass is complete.  Now update parameters
    double m_hat, v_hat;
    for(int i = 0;i < 3*n_s;i++){
      int n = (i < 2*n_s? n_s : n_x);
      for(int j = 0;j < 4*(n+1);j++){
        dE_dW(i,j) /= data.nrows();
        M(i,j) = b1*M(i,j) + (1 - b1)*dE_dW(i,j);
        m_hat = M(i,j)/(1-b1t);
        V(i,j) = b2*V(i,j) + (1-b2)*dE_dW(i,j)*dE_dW(i,j);
        v_hat = V(i,j)*(1/(1-b2t));
        W(i,j) -= dE_dW(i,j)*a*m_hat/(v_hat+eps);
        b1t *= b1;
        b2t *= b2;
      }
    }      
  } // on to next iteration
  cout << "output:\n"<< output;
}
