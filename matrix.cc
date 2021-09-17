#include "Matrix.h"
#include "util.h"

// matrix routines for SCALAR=double

//void printmat(matrix& A){ cout << A;}

bool scale_rows(matrix& C){
  double d;
  for(int i = 0;i < C.nrows();i++){
    if( (d = C(i,i)) != 0){
      for(int j = 0;j < C.ncols();j++) {
        C(i,j) /= d;
      }
    }
    else return false;
  }
  return true;
}



int ut0(matrix& A, double eps){// upper-triangularize in place by Givens row rotations
  int nrot = 0;
  double cos_t, sin_t, h, a,b, tmp;
  for(int i = 1;i < A.nrows();i++){
    for(int j = 0;j < i;j++){
      if(A(i,j) > A(j,j)){ // numerical hygene for h = sqrt(A(i,j)^2 + A(j,j)^2
        a = A(i,j);
        b = A(j,j);
      }
      else{
        a = A(j,j);
        b = A(i,j);
      }
      if(a < eps) continue;
      h = a*sqrt(1+b*b/(a*a));
      nrot++;
      sin_t = A(j,j)/h;
      cos_t = A(i,j)/h;
      A(j,j) = h;
      A(i,j) = 0;
      for(int k = j+1;k < A.ncols();k++){
        tmp = sin_t*A(j,k) + cos_t*A(i,k);
        A(i,k) = -cos_t*A(j,k) + sin_t*A(i,k);
        A(j,k) = tmp;
      }
    }
  }
  return nrot;
}



int ut(matrix& A, double eps){// upper-triangularize in place by Givens row rotations
  int nrot = 0; // eps has a default argument, but eps is not used!
  Givens R;

  for(int i = 1;i < A.nrows();i++){
    for(int j = 0;j < std::min(i,A.ncols());j++){
      if(!R.reset(A(j,j),A(i,j))) continue;
      nrot++;
      for(int k = j+1;k < A.ncols();k++){
        R.rotate(A(j,k),A(i,k));
      }
    }
  }
  return nrot;
}

double trace(const matrix& A){
  double sum = 0;
  int n = A.nrows();
  assert(n == A.ncols());
  for(int i = 0;i < n;i++)sum += A(i,i);
  return sum;
}

double det(const matrix& A){
  assert(A.nrows() == A.ncols());
  matrix B = A.copy();
  ut(B);
  double d = 1.0;
  for(int i = 0;i < B.nrows();i++){
    d *= B(i,i);
    if(d < 1.0e-20)return 0;
  }
  return d;
}

// template<>
// double Matrix<double>::inv(void){
//   assert(_nrows == _ncols);
//   double det = 1.0;
//   ut(*this);
//   for(int i = 0;i < _nrows;i++){
//     det *= ENTRY(i,i);
//     if(det < 1.0e-20){
//       det = 0;
//       break;
//     }
//   }
//   reduce(*this);
//   return det;
// }
    
matrix qr(const matrix& A){
  double zero(0);
  matrix QR(A.nrows(),A.nrows()+A.ncols(),&zero);
  matrix Q(QR.slice(0,A.ncols(),A.nrows(),A.nrows()));
  matrix R(QR.slice(0,0,A.nrows(),A.ncols()));

  R.copy(A);
  for(int i = 0;i < Q.nrows();i++)Q(i,i) = 1.0; // initialize Q to I
  ut(QR);
  return QR;
}


bool reduce(matrix& A, double eps){ // row-reduce upper-triangular A in place
  int n = A.nrows();
  double a,b;

  for(int i = 0;i < n;i++){
    //cout << "reduce:\n"<<A;
    if( fabs(b = A(i,i)) < eps) return false;
    A(i,i) = 1.0;
    for(int j = i+1;j < A.ncols();j++) A(i,j) /= b;
    for(int k = 0;k < i;k++){
      a = A(k,i);
      A(k,i) = 0;
      for(int j = i+1;j < A.ncols();j++) A(k,j) -= a*A(i,j);
      //cout << format("\n(k,i) = (%d,%d) A:\n",k,i)<<A;
    }
  }
  return true;
}

void solve(matrix& A, double eps){ // solve linear equations in-place 
  // A is an m x (m+k) matrix of m equations (in m unknows) with k right-hand sides
  ut(A); // rotate  to upper-triangular form;
  reduce(A,eps); // now row-reduce coefficients to identity.  Each rhs is now solved
}

// matrix inv(const matrix& A, double eps){
//   matrix QR = qr(A); // rotate to upper-triangular form
//   // and append the rotation matrix (as extra columns)
//   int n = QR.nrows();

//   reduce(QR,eps); // now row-reduce
//   return QR.slice(0,n,n,n); // return the left-most n columns
// }

matrix inv(const matrix& A, double* det){ // general matrix inverse
  assert(A.nrows() == A.ncols());
  matrix QR = qr(A); // rotate to upper-triangular form
  // and append the rotation matrix (as extra columns)
  //    cout << "after qr:\n";QR.print();
  int n = QR.nrows();
  matrix Q(QR.slice(0,n,n,n));
  matrix R(QR.slice(0,0,n,n));
  if(det != nullptr){
    *det = 1.0;
    //  cout << "\nR:\n"<<R<<endl;
    for(int i = 0;i < n;i++){
      if(R(i,i) < 0){
        std::cout << "error at i = "<<i<<"R:\n"; R.print();
        std::fflush(stdout);
      }
      if(fabs(R(i,i)) < 1.0e-20){
        *det = 0;
        break;
      }
      *det *= R(i,i);
      if(fabs(*det)< 1.0e-20){
        *det = 0;
        break;
      }
    }
  }
  reduce(QR); // now row-reduce
  return Q;
}

matrix sym_inv(const matrix& A, double* det){ // symmetric matrix inverse
  assert(A.nrows() == A.ncols());
  int n = A.nrows();
  matrix CB(n,2*n);
  matrix C(CB.slice(0,0,n,n));
  matrix B(CB.slice(0,n,n,n));
  cholesky(A,C); // A = C.Tr()*C
  B.fill(0);
  for(int i = 0;i < n;i++) B(i,i) = 1.0; // B = I
  if(det != nullptr){
    double d = 1.0;
    for(int i = 0;i < n;i++){
      if(C(i,i) < 0){
        std::cout << "sym_inv error at i = "<<i<<" C:\n"; C.print();
        std::fflush(stdout);
      }
      d *= C(i,i);
      if(fabs(d)< 1.0e-20){
        d = 0;
        break;
      }
    }
    *det = d*d;
  }
  reduce(CB); // now row-reduce. now B = C_inv
#if 0
  matrix T = A*B*(B.Tr());
  for(int i = 0;i < n;i++){
    for(int j = 0;j < i;j++){
      if(fabs(T(i,j)) > 1.0e-10){
        std::cout << "A:\n"<<A<<"T:\n"<<T;
        std::fflush(stdout);
        throw "oops!";
      }
    }
    if(fabs(T(i,i)-1.0) > 1.0e-10){
      std::cout << "A:\n"<<A<<"T:\n"<<T;
      std::fflush(stdout);
      throw "oops!";
    }
  }
#endif
  return B*(B.Tr()); // A_inv = C_inv*C_inv.Tr()
}

bool cholesky(const matrix& M, matrix& C, double eps){ // user-supplied answer
  /* M is symmetric positive definite. C is upper triangular with C^tC = M.*/
  
  C.fill(0);
  for(int j = 0;j < M.ncols();j++){
    for(int i = 0;i <= j;i++){
      for(int k = 0;k < i;k++) C(i,j) += C(k,i)*C(k,j);
      if(i < j){
        C(i,j) = ((M(i,j)+M(j,i))/2 - C(i,j))/C(i,i);
      }
      else{
        if(M(j,j) - C(j,j) < 0) return false;
        C(i,i) = sqrt(M(j,j)-C(j,j));
        if(C(i,i) < eps) return false; 
      }
    }
  }
  return true;
}

matrix cholesky(const matrix& M){ // we return the answer
  matrix C(M.nrows(),M.ncols());
  cholesky(M,C);
  return C;
}

Array<matrix> svd(const matrix& A, double eps, int maxiters){
  Array<matrix> QR(2);
  Array<matrix> R(2);
  int j,niters,n = std::min(A.nrows(),A.ncols());
  double delta;

  QR[0] = qr(A);
  R[0] = QR[0].slice(0,0,A.nrows(),A.ncols());
  QR[1] = qr(R[0].Tr());
  R[1] = QR[1].slice(0,0,A.ncols(),A.nrows());
  j = 0;
  niters = 0;
  do{
    R[j].copy(R[1-j].Tr());
    ut(QR[j]);
    delta = 0;
    for(int i = 0;i < n;i++){
      if(QR[j](i,i) < eps)continue;
      delta += fabs(QR[j](i,i) - QR[1-j](i,i))/QR[j](i,i);
    }
    delta /= n;
    //    cout << "iteration "<<niters<<": delta = "<<delta<<endl;
    //    cout << format("R[%d]:\n",j)<<R[j];
    j = 1-j;
  }while(delta > eps && ++niters < maxiters);
  return QR;
}


void Svd::reduce(const matrix& A0){
  Givens R;
  reset(A0.nrows(),A0.ncols()); // call this for backward compatibility
  A.copy(A0); // copy in the input
  //cout <<"at line 159:\n"<<AUV;
  ut(AU); // row rotate to upper triangular form
  //cout << "at line 161:\n" << AUV;
  for(int i = 0;i < nrows-1;i++){ // zero the ith row above the super-diagonal with column rotations
    for(int j = ncols-1;j > i+1;j--){ // rotate columns j and j-1 to zero out AV(i,j)
      if(!R.reset(AV(i,j-1),AV(i,j)))continue;
      AV(i,j-1) = R.h; AV(i,j) = 0;
      for(int k = i+1;k < nrows+ncols;k++)
        R.rotate(AV(k,j-1),AV(k,j));
      //      cout << "at line 168:\n" << AUV;
      if(!R.reset(AU(j-1,j-1),AU(j,j-1)))continue;
      // remove resulting lower-triangular "residue" with a row rotation
      AU(j-1,j-1) = R.h;
      AU(j,j-1) = 0;
      for(int k = j;k < nrows+ncols;k++)
        R.rotate(AU(j-1,k),AU(j,k));
    }
  }
  //  cout << "at line 177:\n" << AUV;
  // OK, A is now upper triangular with only the first super-diagonal non-zero
  int  maxiters = 1000;
  bool doit,not_done;
  for(int niters = 0;niters < maxiters;niters++){
    not_done = false;
    for(int j = 0;j < ncols-1;j++){
      if( (doit = R.reset(A(j,j),A(j,j+1))) ){
        for(int k = j+1;k < nrows+ncols;k++)
          R.rotate(A(k,j),A(k,j+1));
      }
      not_done |= doit;
    }
    if(!not_done)break;
    not_done = false;
    for(int i = 0;i < ncols-1;i++){
      if( (doit = R.reset(A(i,i),A(i+1,i))) ){
        for(int k = i+1;k < nrows+ncols;k++)
          R.rotate(A(i,k),A(i+1,k));
      }
      not_done |= doit;
    }
    if(!not_done)break;
    //      cout <<"iteration "<<niters<<"\n"<<A;
  }
}
void Svd::reset(int nr, int nc){
  if(!AUV.initialized() || nrows != nr || ncols != nc){// re-allocate if necessary
    double zero(0);
    nrows = nr;
    ncols = nc;
    AUV.reset(nrows+ncols,nrows+ncols,&zero);
    // define the submatrices
    AU = AUV.slice(0,0,nrows,ncols+nrows);
    AV = AUV.slice(0,0,nrows+ncols,ncols);
    U = AUV.slice(0,ncols,nrows,nrows);
    V = AUV.slice(nrows,0,ncols,ncols);
    A = AUV.slice(0,0,nrows,ncols);
  }
  U.fill(0); V.fill(0);
  for(int i = 0;i < nrows;i++)U(i,i) = 1.0; // set rotation matrices = identity
  for(int i = 0;i < ncols;i++)V(i,i) = 1.0;
}

    

Array<matrix> svd1(const matrix& A, double eps, int maxiters){
  Array<matrix> QR(2);
  Array<matrix> R(2);
  int j,niters,n = std::min(A.nrows(),A.ncols());
  double delta;

  QR[0] = qr(A);
  R[0] = QR[0].slice(0,0,A.nrows(),A.ncols());
  QR[1] = qr(R[0].Tr());
  R[1] = QR[1].slice(0,0,A.ncols(),A.nrows());
  j = 0;
  niters = 0;
  do{
    R[j].copy(R[1-j].Tr());
    ut(QR[j]);
    delta = 0;
    for(int i = 0;i < n;i++){
      if(QR[j](i,i) < eps)continue;
      delta += fabs(QR[j](i,i) - QR[1-j](i,i))/QR[j](i,i);
    }
    delta /= n;
    //    cout << "iteration "<<niters<<": delta = "<<delta<<endl;
    //    cout << format("R[%d]:\n",j)<<R[j];
    j = 1-j;
  }while(delta > eps && ++niters < maxiters);
  return QR;
}

double dot_cols(const matrix& A, int i, int j){
  double ans = 0;
  for(int k = 0;k < A.nrows();k++) ans += A(k,i)*A(k,j);
  return ans;
}
double dotAB(const matrix& A, const matrix& B, int i, int j){
  double ans = 0;
  for(int k = 0;k < A.nrows();k++) ans += A(k,i)*B(k,j);
  return ans;
}

double gram_schmidt(matrix& A, matrix& S, double eps){ // orthonormalize the columns of A 
  int m = A.nrows();
  int n = A.ncols();
  eps = eps*eps;
  double error = 0;

  for(int i = 0;i < n;){
    // inductively, col.s 0,1,...,i-1 are already orthonormal
    S(0,i) = dot_cols(A,i,i); // save initial squared length
    if(S(0,i) > eps){
      for(int j = 0;j < i;j++){
        double scale = dot_cols(A,i,j);
#ifdef VERBOSE
        cout <<format("dot product of col %d = ",i);
        for(int k = 0;k < m;k++)cout << A(k,i)<<" ";
        cout <<format("\n\twith col %d = ",j);
        for(int k = 0;k < m;k++)cout << A(k,j)<<" ";
        cout <<"\nis "<<scale<<endl;
        cout <<"length of col "<<i<<" is "<<sqrt(S(0,i))<<endl;
#endif
        for(int k = 0;k < m;k++)A(k,i) -= scale*A(k,j); // A_i -= <A_i,A_j>A_j
        // OK, col. i is now orthogonal to cols 0,1,...,j
        double e0 = fabs(scale)/sqrt(S(0,i)); // |cos(\theta_i)|
        //        cout << format("e0(%d,%d) = %f\n",i,j,e0);
        if(e0 > error)error = e0;
        S(0,i) -= scale*scale; // adjust sq. length
        //      cout << format("error %d %d = %f\n",i,j,error);
      }
    }
    if(S(0,i) < eps){// swap with last col and reduce n
      if(i < n-1){
        double temp;
        for(int k = 0;k < m;k++){
          temp = A(k,i);
          A(k,i) = A(k,n-1);
          A(i,n-1) = temp;
        }
      }
      S(0,n-1) = 0;
      n--;
      continue;
    }
    S(0,i) = sqrt(S(0,i));
    for(int k = 0;k < m;k++) A(k,i) /= S(0,i); // normalize col i
    i++;
  }
  return error; // max value of |cos(\theta_i)|
}

void print_mat(const matrix& A, int i1, int i2){ std::cout <<"\n"<<A.slice(i1,0,i2-i1+1,A.ncols());}
  
MatrixWelford::MatrixWelford(int d, double t) {reset(d,t);}
void MatrixWelford::reset(int d, double t){
  dim = d;
  W = 0;
  tau = t;
  S1.reset(dim);
  S2.reset(dim,dim);
  S1.fill(0);
  S2.fill(0);
  delta.reset(dim);
}
void MatrixWelford::reset(void){
  W = 0;
  S1.fill(0);
  S2.fill(0);
}
                                                                    
void MatrixWelford::update(const ColVector<double>& x, double weight){
  if(weight < 1.0e-10) return; // effectively zero
  delta = x.copy()*weight;
  if(W > 0) delta = delta - S1*(weight/W);
  W = tau*W + weight;
  S1 = S1*tau + x*weight;
  S2 = S2*tau + delta*(x - S1*(1.0/W)).Tr();
}
