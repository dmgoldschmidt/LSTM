#include <iostream>
#include <cassert>
#include "util.h"
#include "LSTM.h"
using namespace std;

void LSTMcell::static_initialization(int ss, int dd ){
  
  s = ss; // state/output dimension
  d = dd; // input dimension

  //initialize gradients
  dE_dW_xdu.reset(s,d);
  dE_dW_vdu.reset(s,s);
  dE_dW_vcu.reset(s,s);
  dE_dW_xcu.reset(s,d);
  dE_dW_scu.reset(s,s);
  dE_dW_vcs.reset(s,s);
  dE_dW_xcs.reset(s,d);
  dE_dW_scs.reset(s,s);
  dE_dW_xcr.reset(s,d);
  dE_dW_scr.reset(s,s);
  dE_dW_vcr.reset(s,s);

  // initialize backward pass temporaries
  alpha_du.reset(s);alpha_cu.reset(s);alpha_cs.reset(s);alpha_cr.reset(s);d_cu.reset(s);d_cs.reset(s);d_cr.reset(s);
  d_s.reset(s);chi.reset(s);rho.reset(s);
  gamma.reset(s);psi.reset(s); 


}

void LSTMcell::reset(LSTMcell* pc, LSTMcell* nc, const Array<Matrix<double>>& p) :
  W_xdu(p[0]),
  W_vdu(p[1]),
  W_vcu(p[2]),
  W_xcu(p[3]),
  W_scu(p[4]),
  W_vcs(p[5]),
  W_xcs(p[6]),
  W_scs(p[7]),
  W_xcr(p[8]),
  W_scr(p[9]),
  W_vcr(p[10])
{
  prev_cell = pc; next_cell = nc;
  u.reset(s);r.reset(s);a_du.reset(s);a_cu.reset(s);a_cs.reset(s);a_cr.reset(s);
  g_cu.reset(s);g_cs.reset(s);g_cr.reset(s);f_chi.reset(s);f_psi.reset(s);
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
   _state = old_state&g_cs + u&g_cu;
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
  psi = rho&d_s + W_scr*alpha_cr + next_cell->f_psi;
  alpha_cs = psi&prev_cell->state()&d_cs;
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
  
