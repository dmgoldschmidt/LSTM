#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <fenv.h>

#include "util.h"
#include "gzstream.h"
#include "Awk.h"
#include "Matrix.h"
#include "GetOpt.h"
#include "LSTM.h"



// #include "nr3/nr3.h"
// #include "nr3/ran.h"
// #include "nr3/gamma.h"
// #include "nr3/deviates.h"
// #include "nr3/erf.h"


using namespace std;

void ranfill(Matrix<double>& M, Normaldev& gen){
  for(int i = 0;i < M.nrows();i++){
    for(int j = 0;j < M.ncols();j++) M(i,j) = gen.dev();
  }
}

int main(int argc, char** argv){
  int n_x = 1; // data dimension
  int n_s = 1; // state-output dimension
  int ndata = 1; // no. of data points
  int seed = 12345;
  
  GetOpt cl(argc,argv);
  cl.get("n_x",n_x);
  cl.get("n_s",n_s);
  cl.get("ndata",ndata);
  cl.get("seed",seed);
  
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  Normaldev gen(0,1,seed);
  int n = max(n_s,n_x);
  Matrix<double> W(3*n_s,4*(n+1)), dE_dW(3*n,4*(n+1));
  ranfill(W,gen);
  dE_dW.fill(0);
  ColVector<double> v(n_s+1);
  ColVector<double> s(n_s+1);
  RowVector<double> dE_ds(n_s);
  RowVector<double> dE_dv(n_s);
  v.fill(0);
  s.fill(0);
  dE_ds.fill(0);
  dE_dv.fill(0);
  
  Matrix<double> data(n_x,ndata+1);
  for(int t = 1;t <= ndata;t++){
    for(int i = 0;i < n_x;i++){
      data(i,t) = gen.dev();
      cout << format("data(%d,%d) = %.3f\n",i,t,data(i,t));
    }
  }
  RowVector<double> dE_dxn(1);
  dE_dxn[0] = 1.0;
  Cell cell(1,1,v,s,W,dE_dW,dE_dv,dE_ds);
  ColVector<double> x = {1,gen.dev()};
  cout << "input: "<<x<<endl;
  cell.forward_step(x);
  cout << "after forward_step:\n"<<cell;
  cell.backward_step(dE_dxn);
  cout << "after backward_step:\n"<<cell;
}
  // Matrix<double> output(n_s,ndata+1);
  // LSTM lstm(data,output,W);
  // for(int t = 1; t <= n_data;t++){
  //   ColVector<double> x = data.slice(0,t,n_x,1);
  //   //    ColVector<double> y = output.slice(0,t,n_s,1);
  //   lstm.cells[1].forward_step(x);
  //   lstm_1.cells[1].forward_step(x);
  //   output.slice(0,t,n_s,1).copy(lstm.cells[1].readout());
  //   output_1.slice(0,t,n_s,1).copy(lstm_1.cells[1].readout());
  // }
  // cout<<"data:\n"<<data<<"output:\n"<<output<<"output_1:\n"<<output_1;

    
                            


