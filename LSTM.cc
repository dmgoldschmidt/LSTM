#include <iostream>
#include <cassert>
#include "util.h"
#include "LSTM.h"
using namespace std;

ColVector<double> squash(ColVector<double> x){
  for(int i = 0;i < x.nrows();i++)x[i] = sigma(x[i]);
  return x;
}


void G_c(ColVector<double>& x, ColVector<double>& g){
  assert(g.nrows() == x.nrows());
  int dim = x.nrows();
  for(int i = 0; i < dim;i++) g[i] = sigma(x[i]);
}
void dG_c(ColVector<double>& x, ColVector<double>& g){
  assert(g.nrows() == x.nrows());
  int dim = x.nrows();
  for(int i = 0; i < dim;i++)g[i] = sigma(x[i])*sigma(-x[i]);
}
void G_d(ColVector<double>& x, ColVector<double>& g){
  assert(g.nrows() == x.nrows());
  int dim = x.nrows();
  for(int i = 0;i < dim;i++) g[i] = 2*sigma(x[i])-1;
}
void dG_d(ColVector<double>& x, ColVector<double>& g){
  assert(g.nrows() == x.nrows());
  int dim = x.nrows();
  for(int i = 0;i < dim;i++)g[i] = 2*sigma(x[i])*sigma(-x[i]);
}

//static variables defined at global scope

int LSTMcell::n_s; // dimension of the state and output vectors
int LSTMcell::n_x; // dimension of the input vector

  
/* backward pass temporaries
 * they are esssentially scratch computations that don't need to be saved
 * so they are static to minimize dynamic memory thrashing. Each instance
 * overwrites whatever is there from the last time step.
 *
 */
ColVector<double> LSTMcell::alpha_du; 
ColVector<double> LSTMcell::alpha_cu;
ColVector<double> LSTMcell::alpha_cs;
ColVector<double> LSTMcell::alpha_cr;

ColVector<double> LSTMcell::d_cu; // G_c'(a_cu)
ColVector<double> LSTMcell::d_cs; // G_c'(a_cs)
ColVector<double> LSTMcell::d_cr; // G_c'(a_cr)
ColVector<double> LSTMcell::d_s; // G_d'(state)
ColVector<double> LSTMcell::d_du; // G_d'(a_du)

ColVector<double> LSTMcell::chi; // dE/dv
ColVector<double> LSTMcell::rho; // dE/dr
//ColVector<double> LSTMcell::gamma; // dE/dg
ColVector<double> LSTMcell::psi; // dE/ds

/* these are the actual gradients we will use to update parameters for
 * the next iteration.  Each cell object adds its contribution.
 */
Matrix<double> LSTMcell::dE_dW_xdu;
Matrix<double> LSTMcell::dE_dW_vdu;
Matrix<double> LSTMcell::dE_dW_vcu;
Matrix<double> LSTMcell::dE_dW_xcu;
Matrix<double> LSTMcell::dE_dW_scu;
Matrix<double> LSTMcell::dE_dW_vcs;
Matrix<double> LSTMcell::dE_dW_xcs;
Matrix<double> LSTMcell::dE_dW_scs;
Matrix<double> LSTMcell::dE_dW_xcr;
Matrix<double> LSTMcell::dE_dW_scr;
Matrix<double> LSTMcell::dE_dW_vcr;

void LSTMcell::static_initializer(int ss, int xx){
  
  n_s = ss; // state dimension
  n_x = xx; // input dimension

  //initialize gradients
  dE_dW_xdu.reset(n_s,n_x); 
  dE_dW_xcu.reset(n_s,n_x);
  dE_dW_xcs.reset(n_s,n_x);
  dE_dW_xcr.reset(n_s,n_x);
  dE_dW_vcr.reset(n_s,n_s);
  dE_dW_scr.reset(n_s,n_s);
  dE_dW_vdu.reset(n_s,n_s);
  dE_dW_vcu.reset(n_s,n_s);
  dE_dW_vcs.reset(n_s,n_s);
  dE_dW_scu.reset(n_s,n_s);
  dE_dW_scs.reset(n_s,n_s);

  // initialize backward pass temporaries
  alpha_du.reset(n_s);alpha_cu.reset(n_s);alpha_cs.reset(n_s);alpha_cr.reset(n_s);d_cu.reset(n_s);
  d_cs.reset(n_s);d_cr.reset(n_s);d_s.reset(n_s);d_du.reset(n_s);chi.reset(n_s);rho.reset(n_s);psi.reset(n_s); 
}

void LSTMcell::reset(LSTMcell* pc, LSTMcell* nc, Array<Matrix<double>>& p)
{
  W_xdu = p[0];  //NOTE:  these are shallow copies.  There's only one actual set of parameters
  W_xcu = p[1];
  W_xcs = p[2];
  W_xcr = p[3];
  W_vcr = p[4];
  W_scr = p[5];
  W_vdu = p[6];
  W_vcu = p[7];
  W_vcs = p[8];
  W_scu = p[9];
  W_scs = p[10];

  prev_cell = pc; next_cell = nc;_state.reset(n_s);_readout.reset(n_s),_input.reset(n_x);
  u.reset(n_s);r.reset(n_s);a_du.reset(n_s);a_cu.reset(n_s);a_cs.reset(n_s);a_cr.reset(n_s);
  g_cu.reset(n_s);g_cs.reset(n_s);g_cr.reset(n_s);f_chi.reset(n_s);f_psi.reset(n_s);
}


void LSTMcell::forward_step(ColVector<double>& x){
   _input.copy(x); // save for backward step
   assert(prev_cell != nullptr);
   ColVector<double>& v = prev_cell->_readout;
   ColVector<double>& old_state = prev_cell->_state;
   a_cu = W_xcu*x + W_scu*old_state + W_vcu*v;
   a_cs = W_xcs*x + W_scs*old_state + W_vcs*v;
   a_du = W_xdu*x + W_vdu*v;
   G_c(a_cu,g_cu);
   G_c(a_cs,g_cs);
   G_d(a_du,u); 
   _state = (old_state&g_cs) + (u&g_cu);
   a_cr = W_xcr*x + W_scr*_state + W_vcr*v;
   G_c(a_cr,g_cr);
   G_d(_state,r);
   _readout = g_cr&r;
}

void LSTMcell::backward_step(ColVector<double>& dE_dv){
  assert(next_cell != nullptr && prev_cell != nullptr);
  dG_c(a_cr,d_cr);
  dG_c(a_cs,d_cs);
  dG_c(a_cu,d_cu);
  dG_d(a_du,d_du);
  dG_c(_state,d_s);
  chi = dE_dv + next_cell->f_chi;
  rho = chi&g_cr;
  alpha_cr = chi&r&d_cr;
  psi = (rho&d_s) + W_scr*alpha_cr + next_cell->f_psi;
  alpha_cs = (prev_cell->state())&d_cs&psi;
  //  alpha_cs = alpha_cs&d_cs;
  alpha_cu = psi&u&d_cu;
  alpha_du = psi&g_cu&d_du;
  f_chi = W_vcu*alpha_cu + W_vcs*alpha_cs + W_vcr*alpha_cr + W_vdu*alpha_du;
  f_psi = W_scu*alpha_cu + W_scs*alpha_cs + (g_cs&psi);

  RowVector<double> v = prev_cell->_readout.T();
  RowVector<double> x = _input.T();
  RowVector<double> s0 = prev_cell->_state.T();
  RowVector<double> s1 = _state.T();
  dE_dW_xdu += alpha_du*x;
  dE_dW_vdu += alpha_du*v;
  dE_dW_vcu += alpha_cu*v;
  dE_dW_xcu += alpha_cu*x;
  dE_dW_scu += alpha_cu*s0;
  dE_dW_vcs += alpha_cs*v;
  dE_dW_xcs += alpha_cs*x;
  dE_dW_scs += alpha_cs*s0;
  dE_dW_xcr += alpha_cr*x;
  dE_dW_scr += alpha_cr*s1;
  dE_dW_vcr += alpha_cr*v;
}
  
LSTM::LSTM(Matrix<double>& d, Matrix<double>& o, Array<Matrix<double>>& p) :
  data(d),output(o),parameters(p){

  LSTMcell::static_initializer(output.nrows(),data.nrows());
  for(int i = 0;i < 4;i++)assert(data.nrows() == parameters[i].ncols()); // input dimension
  for(int i = 4;i < 10;i++)assert(output.nrows() == parameters[i].ncols());
  for(int i = 0;i < 10;i++)assert(output.nrows() == parameters[i].nrows());

  int& n_s = LSTMcell::n_s;
  ncells = data.ncols(); 
  cells.reset(ncells+2); // first and last cell are initializers

  cells[0].reset(nullptr,&cells[1],parameters);
  cells[ncells+1].reset(&cells[ncells],nullptr,parameters);
  for(int i = 1;i <= ncells;i++)cells[i].reset(&cells[i-1],&cells[i+1],parameters);
  for(int i = 0;i <  n_s;i++){
    cells[0]._state[i] = cells[0]._readout[i] = 0;
    cells[ncells+1].f_chi[i] = cells[ncells+1].f_psi[i] = 0;
  }
}

void LSTM::train(int niters,double pct,double learn,double eps){
}
