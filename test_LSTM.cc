#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <fenv.h>

#include "util.h"
#include "stats.h"

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


//using namespace std;

void ranfill(Matrix<double>& M, Normaldev& gen){
  for(int i = 0;i < M.nrows();i++){
    for(int j = 0;j < M.ncols();j++) M(i,j) = gen.dev();
  }
}

int main(int argc, char** argv){
  int n_x = 2; // data dimension
  int n_s = 2; // state-output dimension
  int ndata = 20; // no. of data points
  int seed = 12345;
  int ncells = 0;
  int niters = 100;
  double alpha = .1; // parameter correction step_size
  
  GetOpt cl(argc,argv);
  cl.get("n_x",n_x);
  cl.get("n_s",n_s);
  cl.get("ndata",ndata);
  cl.get("seed",seed);
  cl.get("ncells",ncells);
  cl.get("niters",niters);
  cl.get("alpha",alpha);
  
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  Normaldev normal(0,1,seed);
  int n = std::max(n_s,n_x);
  Matrix<double> W(3*n_s,4*(n+1)); // n+1 leaves room for bias
  ranfill(W,normal);
  cout << "initial parameters:\n"<<W;
  Ran random(seed);
  //  Array<int> data0(ndata+n_x+n_s);
  Array<ColVector<double>> data(ndata);
  cout << "data:\n";
  for(int t = 0;t < ndata;t++){
    data[t].reset(n_x+1);
    data[t][0] = 1;  // augmentation
    data[t][t%2 + 1] = 1; // 1-hot revs
    data[t][(t+1)%2 + 1] = 0;
    cout << data[t].Tr();
  }
  Matrix<double> output(ndata,n_s);
  LSTM lstm(n_s,n_x,ncells,data,output,W);
  lstm.train(niters,alpha);
  cout << "updated parameters:\n"<<W;
}


    
                            


