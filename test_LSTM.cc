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

int main(int argc, char** argv){
  int n_x = 1; // data dimension
  int n_s = 1; // state-output dimension
  int n_data = 1; // no. of data points
  int seed = 12345;
  
  GetOpt cl(argc,argv);
  cl.get("n_x",n_x);
  cl.get("n_s",n_s);
  cl.get("n_data",n_data);
  cl.get("seed",seed);
  
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  Normaldev gen(0,1,seed);
  Array<Matrix<double>> parameters(11);
  for(int k = 0;k < 4;k++)parameters[k].reset(n_s,n_x);
  for(int k = 4;k < 11;k++)parameters[k].reset(n_s,n_s);
  for(int k = 0;k < 11;k++){ // set all parameters to random values
    for(int i = 0;i < parameters[k].nrows();i++){
      for(int j = 0;j < parameters[k].ncols();j++) parameters[k](i,j) = gen.dev();
    }
  }
  Matrix<double> data(n_x,n_data);
  for(int t = 0;t < n_data;t++){
    for(int i = 0;i < n_x;i++) data(i,t) = gen.dev();
  }
  Matrix<double> output(n_s,n_data+1);
  LSTM lstm(data,output,parameters);
  ColVector<double> x(n_x),y(n_s);
  for(int t = 1; t <= n_data;t++){
    x = data.slice(0,t,1,n_x);
    y = output.slice(0,t,1,n_s);
    lstm.cells[1].forward_step(x);
    y = lstm.cells[1].readout();
  }
  cout << "x: "<<x.T()<<endl;
  cout << "y: "<<y.T()<<endl;
}
    
                            


