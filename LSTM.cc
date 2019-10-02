#include <iostream>
#include <cassert>
#include "util.h"
#include "LSTM.h"
using namespace std;

static void LSTMcell::static_initialization
(
 const Matrix<double>& W_xdu0,
 const Matrix<double>& W_vdu0,

 const Matrix<double>& W_vcu0,
 const Matrix<double>& W_xcu0,
 const Matrix<double>& W_scu0,

 const Matrix<double>& W_vcs0,
 const Matrix<double>& W_xcs0,
 const Matrix<double>& W_scs0,

 const Matrix<double>& W_xcr0,
 const Matrix<double>& W_scr0,
 const Matrix<double>& W_vcr0){

  //initialize parameters
  W_xdu.copy(W_xdu0);
  W_vdu.copy(W_vdu0);
  W_vcu.copy(W_vcu0);
  W_xcu.copy(W_xcu0);
  W_scu.copy(W_scu0);
  W_vcs.copy(W_vcs0);
  W_xcs.copy(W_xcs0);
  W_scs.copy(W_scs0);
  W_xcr.copy(W_xcr0);
  W_scr.copy(W_scr0);
  W_vcr.copy(W_vcr0);

  
  // OK, static model parameters are set
  
  s = W_xdu0.nrows(); // state/output dimension
  d = W_xdu0.ncols(); // input dimension

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
  d_r.reset(s);d_u.reset(s);chi.reset(s);rho.reset(s);
  gamma.reset(s);psi.reset(s); 


}

void LSTMcell::LSTMcell(void) :
  prev_cell(pc), next_cell(nc),u(s),r(s),a_du(s),a_cu(s),a_cs(s),a_cr(s),
  g_cu(s),g_cs(s),g_cr(s),f_chi(s),f_psi(s) {}


 void LSTMcell::forward_step(ColVector<double>& x){
   ColVector<double>& old_state = (prev_cell != nullptr? prev_cell->_state : 0);
   ColVector<double>& new_state = _state;
   ColVector<double>& v = (prev_cell != nullptr? prev_cell->_readout : 0);
   _input.copy(x); // save for backward step

   a_cu = W_xcu*x + W_scu*old_state + W_vcu*v; 
   a_cs = W_xcs*x + W_scs*old_state + W_vcs*v;
   G_c(a_cu,g_cu);
   G_c(a_cs,g_cs);
   G_d(W_xdu*x + W_vdu*v,u); 
   new_state = old_state&g_cs + u&g_cu;
   a_cr = W_xcr*x + W_scr*new_state + W_vcr*v;
   G_c(a_cr,g_cr);
   G_d(_state,r);
   _readout = g_cr&r;
 }
void LSTMcell::backward_step(ColVector<double>& dE_dv){
  dG_c(a_cr,d_cr);
  dG_c(a_cs,d_cs);
  dG_c(a_cu,d_cu);
  dG_d(_state,d_du);
  chi = dE_dv + (next_cell != nullptr? next_cell->f_chi() : 0);
  rho = chi&g_cr;
  alpha_cr = chi&r&d_cr;
  psi = rho&d_r&W_scr*alpha_cr + (next_cell != nullptr? next_cell->f_psi() : 0);
  alpha_cs = psi&prev_cell->state()&d_cs;
  alpha_cu = psi&u&d_cu;
  alpha_du = psi&g_cu&d_du;
  f_chi = W_vcu*alpha_cu + W_vcs*alpha_cs + W_vcr*alpha_cr + W_vdu*alpha_du;
  f_psi = W_scu*alpha_cu + W_scs*alpha_cs + g_cs&psi;

  RowVector<double> v = prev_cell->_readout.T();
  RowVector<double> x = input.T();
  RowVector<double> s0 = prev_cell->_state.T();
  RowVector<double> s1 = _state.T();
  dE_dW_xdu += alpha_du*x.T();
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
  
