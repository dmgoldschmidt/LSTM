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
  int n_x = 4; // data dimension
  int n_s = 1; // state-output dimension
  int ndata = 20; // no. of data points
  int seed = 12345;
  int ncells = 0;
  int niters = 10;
  double alpha = .001; // parameter correction step_size
  
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
  Array<int> data0(ndata+n_x+n_s);
  Matrix<double> data(ndata,n_x+n_s);
  for(int t = 0;t < ndata+n_x+n_s;t++){
    data0[t] = 2*(t%2) - 1;//(random.doub() < .5? -1 : 1);
  }
  for(int t = 0;t < ndata;t++){
    for(int i = 0;i < n_x+n_s;i++){
      data(t,i) = data0[t+i];
    }
  }
  cout << "data0: "<< data0 << endl;
  cout << "data:\n"<<data;
  Matrix<double> output(ndata,n_s);
  LSTM lstm(n_s,n_x,ncells,data,output,W);
  lstm.train(niters,alpha);
  cout << "updated parameters:\n"<<W;
  cout << "output:\n"<<output;
}


    
                            


