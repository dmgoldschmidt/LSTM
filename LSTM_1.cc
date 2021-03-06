#include <iostream>
#include <cassert>
#include "util.h"
#include "LSTM_1.h"
using namespace std;

//static variables defined at global scope

int LSTM_1cell::n_s; // dimension of the state and output vectors
int LSTM_1cell::n_x; // dimension of the input vector

void LSTM_1cell::static_initializer(int ss, int xx){
  n_s = ss; // state dimension
  n_x = xx; // input dimension
  zero.reset(n_s);
  for(int i = 0;i < n_s;i++)zero[i] = 0;
}

ColVector<double> LSTM_1cell::zero;

double LSTM_1cell::sigma(double x){return 1/(1+exp(-x));}

ColVector<double> LSTM_1cell::squash(ColVector<double> x){
  for(int i = 0;i < x.nrows();i++)x[i] = sigma(x[i]);
  return x;
}

ColVector<double> LSTM_1cell::bulge(ColVector<double> x){
  for(int i = 0;i < x.nrows();i++) x[i] =2*x[i]-1;
  return x;
}

void Gate::reset(RowVector<Matrix<double>> W0){
  assert(W0.dim() == 3);
  W = W0; // shallow copy
  //  cout <<"Gate reset: W:\n"<<W;
  for(int i = 0;i < 2;i++)assert(W[i].ncols() == LSTM_1cell::n_s);
  assert(W[2].ncols() == LSTM_1cell::n_x);
  g.reset(LSTM_1cell::n_s);
  dg_dw.reset(3);
  dg_dW.reset(3);
  for(int k = 0;k < 3;k++){
    dg_dw[k].reset(W[k].nrows(),W[k].ncols());
    dg_dW[k].reset(W[k].nrows(),W[k].ncols());
  }
}

ColVector<double> Gate::operator()(RowVector<ColVector<double>>& w){
  // cout << "W[2]: "<<W[2]<<" x: "<<x<<" W[2]*x: "<<W[2]*x;
  // cout << "W[1]: "<<W[1]<<" s: "<<s<<" W[1]*s: "<<W[1]*s;
  // cout << "W[0]: "<<W[0]<<" v: "<<v<<" W[0]*v: "<<W[0]*v;
  ColVector<double>& v = w[0];
  ColVector<double>& s = w[1];
  ColVector<double>& x = w[2];

  g = W[0]*v + W[1]*s + W[2]*x;
  //  cout << "g: "<<g;
  for(int k = 0;k < W.dim();k++){
    for(int i = 0;i < g.nrows();i++){
      double d_sigma_i = LSTM_1cell::sigma(g[i])*(1-LSTM_1cell::sigma(g[i]));
      for(int j = 0;j < W[k].ncols();j++){
        dg_dw[k](i,j) = d_sigma_i*W[k](i,j);
        dg_dW[k](i,j) = d_sigma_i*w[k][j];
      }
    }
  }
  return LSTM_1cell::squash(g);
}




void LSTM_1cell::reset(LSTM_1cell* pc, LSTM_1cell* nc, Matrix<Matrix<double>> W0){
  assert(W0.nrows() == 4 && W0.ncols() == 3);
  gate.reset(4);
  RowVector<Matrix<double>> R(3);
  for(int i = 0;i < 4;i++){
    R = W0.slice(i,0,1,3);
    gate[i].reset(R);
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

void LSTM_1cell::forward_step(ColVector<double>& x){
  w[2].copy(x); // save for backward step
  assert(prev_cell != nullptr);
  w[0].copy(prev_cell->w[0]);
  w[1].copy(prev_cell->w[1]);
  //cout << "\ngate[0].W:\n"<<gate[0].W;
  gate[0](w);
  LSTM_1cell::bulge(gate[0].g); // convert to tanh(2x);
  //cout << "\ngate[1].W:\n"<<gate[1].W;
  gate[1](w);
  //cout << "\ngate[2].W:\n"<<gate[2].W;
  gate[2](w);
  w[1] = (gate[0].g&gate[1].g) + (w[1]&gate[2].g); // update the state
  //cout << "\ngate[3].W:\n"<<gate[3].W;
  gate[3](w); 
  r = bulge(squash(w[1])); // tanh(s/2)
  w[0] = r&gate[3].g;
  //cout << "new v: "<<v<<endl;
}

// void LSTM_1cell::backward_step(RowVector<double>& dE_dv){
//   assert(next_cell != nullptr && prev_cell != nullptr);
//   d_dg = d_dv&r;
//   for(int k = 0;k < 3;k++){
//     (d_dg*gate[3].dg_dw[k]




// }
  
LSTM_1::LSTM_1(Matrix<double>& d, Matrix<double>& o, Matrix<Matrix<double>>& p) :
  data(d),output(o),parameters(p) {
  
  LSTM_1cell::static_initializer(output.nrows(),data.nrows());
  int& n_s = LSTM_1cell::n_s;
  for(int i = 0;i < 4;i++){
    assert(data.nrows() == parameters(i,2).ncols() && data.nrows() == LSTM_1cell::n_x); // input dimension
    for(int j = 0;j < 2;j++)assert(output.nrows() == parameters(i,j).ncols() && output.nrows() == n_s);
  }
  ncells = data.ncols(); 
  cells.reset(ncells+2); // first and last cell are initializers

  cells[0].reset(nullptr,&cells[1],parameters);
  cells[ncells+1].reset(&cells[ncells],nullptr,parameters);
  for(int i = 1;i <= ncells;i++)cells[i].reset(&cells[i-1],&cells[i+1],parameters);
  // initialize first and last cells
  cells[0].w[0].copy(LSTM_1cell::zero);
  cells[0].w[1].copy(LSTM_1cell::zero);
  cells[ncells+1].d_ds.copy(LSTM_1cell::zero);
  cells[ncells+1].d_dv.copy(LSTM_1cell::zero);
}

void LSTM_1::train(int niters,double pct,double learn,double eps){
}
