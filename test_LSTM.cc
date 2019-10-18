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
#include "LSTM_1.h"


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
      for(int j = 0;j < parameters[k].ncols();j++){
        parameters[k](i,j) = gen.dev();
        cout << format("param[%d](%d,%d) = %f\n",k,i,j,parameters[k](i,j));
      }        
    }
  }
  Matrix<Matrix<double>> parameters_1(4,3);
  int k = 0;
  for(int i = 0;i < 4;i++) parameters_1(i,2) = parameters[k++];
  for(int j = 0;j < 2;j++) parameters_1(3,j) = parameters[k++];
  for(int i = 0;i < 3;i++) parameters_1(i,0) = parameters[k++];
  for(int i = 1;i < 3;i++) parameters_1(i,1) = parameters[k++];
  parameters_1(0,1).reset(1,1);
 
  
  Matrix<double> data(n_x,n_data+1);
  for(int t = 1;t <= n_data;t++){
    for(int i = 0;i < n_x;i++){
      data(i,t) = gen.dev();
      cout << format("data(%d,%d) = %f\n",i,t,data(i,t));
    }
  }
  Matrix<double> output(n_s,n_data+1);
  Matrix<double> output_1(n_s,n_data+1);
  LSTM lstm(data,output,parameters);
  LSTM_1 lstm_1(data,output_1,parameters_1);
  for(int t = 1; t <= n_data;t++){
    ColVector<double> x = data.slice(0,t,n_x,1);
    //    ColVector<double> y = output.slice(0,t,n_s,1);
    lstm.cells[1].forward_step(x);
    lstm_1.cells[1].forward_step(x);
    output.slice(0,t,n_s,1).copy(lstm.cells[1].readout());
    output_1.slice(0,t,n_s,1).copy(lstm_1.cells[1].readout());
  }
  cout<<"data:\n"<<data<<"output:\n"<<output<<"output_1:\n"<<output_1;
}
    
                            


