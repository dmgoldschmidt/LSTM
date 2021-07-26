#include <iostream>
#include <cassert>
#include "util.h"
#include "LSTM.h"
using namespace std;

//static variables defined at global scope

// int LSTMcell::n_s; // dimension of the state and output vectors
// int LSTMcell::n_x; // dimension of the input vector
// ColVector<double> LSTMcell::zero;
// RowVector<double> LSTMcell::d_dg;

// void LSTMcell::static_initializer(int ss, int xx){
//   n_s = ss; // state dimension
//   n_x = xx; // input dimension
//   d_dg.reset(n_s);
//   zero.reset(n_s);
//   for(int i = 0;i < n_s;i++)zero[i] = 0;
// }

// double LSTMcell::sigma(double x){return 1/(1+exp(-x));}


void Gate::reset(Matrix<Matrix<double>>& W0, int gn){
  gate_no = gn;
  W = W0; // shallow copy NOTE: W0 will get updated each iteration
  //  cout <<"Gate reset: W:\n"<<W;
  for(int i = 0;i < 2;i++){
    assert(W[i].ncols() == LSTMcell::n_s);
  }
  assert(W[2].ncols() == LSTMcell::n_x);
  g.reset(4);
  dg_dz.reset(3);
  dg_dW.reset(3);
  for(int k = 0;k < 3;k++){
    dg_dz[k].reset(W[k].nrows(),W[k].ncols());
    dg_dW[k].reset(W[k].nrows(),W[k].ncols());
  }
}

ColVector<double> Gate::operator()
  (RowVector<ColVector<double>>& z, ColVector<double>& x){
  int j = gate_no;
  ColVector<double>& v = z[0]; // readout from previous cell
  ColVector<double>& s = z[1]; // state from previous cell

  g = W(0,j)*v + W(2,j)*x;
  if(gate_no > 0) g += W(1,j)*s; // apply weights to the input
  return squash(g);
  // pre-compute stuff for backward pass
  for(int k = 0;k < 3;k++){
    for(int i = 0;i < g.nrows();i++){
      double d_sigma_i = sigma(g[i])*(1 - sigma(g[i]));
      for(int j = 0;j < W[k].ncols();j++){
        dg_dz[k](i,j) = d_sigma_i*W[k](i,j);
        dg_dW[k](i,j) = d_sigma_i*w[k][j];
      }
    }
  }
  return LSTMcell::squash(g);
}




void LSTMcell::reset(LSTMcell* pc, LSTMcell* nc,
                     Matrix<Matrix<double>>& W0){
  assert(W0.nrows() == 3 && W0.ncols() == 4);
  gate.reset(4);
  Array<Matrix<double> R(3);

  for(int j = 0;j < 4;j++){
    for(int i = 0;i < 3;i++) R[i] = W0(i,j); 
    gate[j].reset(R,j);
  }
  w.reset(3);
  w[0].reset(n_s);
  w[1].reset(n_s);
  w[2].reset(n_x);
  r.reset(n_s);
  //  input.reset(n_x);
  
  prev_cell = pc;
  next_cell = nc;
  d_ds.reset(n_s);
  d_dv.reset(n_s);
}

void printmat(Matrix<double>& M){cout << M;}

void forward_step(Array<ColVector<double>>& z, ColVector<double>& x){
  z[1] = gate[2](z,x)&z[1] + gate[0](z,x)&gate[1](z,x);
  z[0] = gate[3](z,x)&bulge(z[1]);
}



// void LSTMcell::backward_step(RowVector<double>& dE_dv){
//   assert(next_cell != nullptr && prev_cell != nullptr);
//   d_dv = next_cell->d_dv + dE_dv;  // input from local error and backprop
//   d_dg = d_dv&r;
  
//   for(int k = 0;k < 3;k++){
//     (d_dg*gate[3].dg_dW[k]




// }
  
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
