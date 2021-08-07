#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <atomic>
#include "using.h"
#include "Array.h"
#include "util.h"

//#define DEBUG 1

#define ENTRY (operator())

template<typename SCALAR>
class Matrix;
typedef Matrix<double> matrix;
typedef Matrix<int> imatrix;
class SparseMatrix;
template<typename SCALAR>
class ColVector;
template<typename SCALAR>
class RowVector;

struct Givens{ // reset computes, applies, and stores the 2x2 rotation matrix T s.t. T(a,b) = (sqrt(a^2+b^2),0) 
  // rotate applies the above rotation to an arbitrary vector (x,y).
  double sin_t;
  double cos_t;
  double h;
  double eps;

  Givens(double e = 1.0e-5) : eps(e){}
  
  bool reset(double& a0, double& b0){
    if(fabs(b0) < eps) return false;
    double a = a0*a0;
    double b = b0*b0;
    h = fabs(b < a? a0*sqrt(1+b/a) : b0*sqrt(1+a/b)); // numerical hygene for sqrt(a+b)
    cos_t = a0/h;
    sin_t = b0/h;
    a0 = h;
    b0 = 0;
    //    std::cout << "sin_t = "<<sin_t<<", cos_t = "<<cos_t<<std::endl;
    return true;
  }
   
  inline void rotate(double& x, double& y){
    double z = x;
    x = cos_t*z + sin_t*y;
    y = -sin_t*z + cos_t*y;
  }
};
template<typename T>
class MemoryBlock {
  int n;
  uint* ref_count;
  T* data;
  inline void inc_ref_count(void){
#pragma omp atomic update
    ++(*ref_count);
  }
  inline void dec_ref_count(void){
#pragma omp atomic update
    --(*ref_count);
  }
public:
  MemoryBlock(void){
    ref_count = nullptr;
    data = nullptr;
    n = 0;
  }
  MemoryBlock(uint size){
    ref_count = new uint;
    *ref_count = 1;
    data = new T[size];
#ifdef DEBUG
    dump("MemoryBlock constructor");
#endif
    n = size;
  }
  ~MemoryBlock(void){
    if(data == nullptr){
      assert(ref_count == nullptr);
      return;
    }
#ifdef DEBUG
      dump("MemoryBlock destructor(entry)");
#endif
    dec_ref_count();
    if(*ref_count <= 0){
      delete ref_count;
      delete[] data;
    }
#ifdef DEBUG
    dump("MemoryBlock destructor(exit)");
#endif
  }
  MemoryBlock(const MemoryBlock& mb){
    data = mb.data;
    ref_count = mb.ref_count;
    inc_ref_count();
    n = mb.n;
#ifdef DEBUG
    dump("MemoryBlock copy constructor(exit)");
#endif    
  }
  
  void reset(uint size){
#pragma omp critical (MemoryBlock_reset)
    {
      if(data != nullptr){
        assert(ref_count != nullptr);
        dec_ref_count();
        if(*ref_count <= 0){
          delete[] data;
          delete ref_count;
        }
      }
      ref_count = new uint;
      *ref_count = 1;
      data = new T[size];
      n = size;
#ifdef DEBUG
    dump("MemoryBlock reset(exit)");
#endif    
    }// end critical region
  }

  MemoryBlock& operator=(const MemoryBlock& mb) {
#pragma omp critical (MemoryBlock_equals)
    {
      if(ref_count == nullptr || data == nullptr) {
        assert(data == nullptr && ref_count == nullptr);
        data = mb.data;
        ref_count = mb.ref_count;
        inc_ref_count();
        n = mb.n;
      }
      else if(data != mb.data){ // we're switching memory blocks
        dec_ref_count();
        if(*ref_count <= 0){
#ifdef DEBUG
          dump("MemoryBlock operator= (deletion)");
#endif
          delete[] data;
          delete ref_count;
        }  
        data = mb.data;
        ref_count = mb.ref_count;
        n = mb.n;
        inc_ref_count();
      }
      else assert(ref_count == mb.ref_count);
#ifdef DEBUG
      dump("MemoryBlock operator= (exit)");
#endif
    } // end critical region
    return *this;
  }
  
  T& operator[](int i) const {return data[i];}
  int len(void) const {return n;}
  bool initialized(void){return data != nullptr;}
  void dump(char* msg = ""){
    cout << format("%s: data(%x) ref_count(%x) = %d\n",msg,data,ref_count,
                   *ref_count);
  }
};
// template<typename T>
// struct MemoryBlock {
//   uint ref_count;
//   T* data;
//   MemoryBlock(size){
//     ref_count = 1;
//     data = new T[size];
//   }
//   ~MemoryBlock(void){delete[] data;}
// };

template<typename SCALAR>
class Matrix {
protected:
  MemoryBlock<SCALAR> data;
  int offset; // these parameters support strides
  int _nrows;
  int _ncols;
  int col_stride;
public:
  bool printall;
  Matrix(void) : offset(0), _nrows(0), _ncols(0), col_stride(0), printall(false) {}
  Matrix(int m, int n, const SCALAR* f = nullptr): data(m*n), offset(0), _nrows(m), _ncols(n), col_stride(n),printall(false){
    if(f != nullptr)fill(*f);
  }

  Matrix(std::initializer_list<Array<SCALAR>> l) : printall(false){
    
    _nrows = l.size();
    _ncols = 0;
    const Array<SCALAR>* p = l.begin();
    
    for(int i = 0;i < _nrows;i++){
      if(p[i].len() > _ncols)_ncols = p[i].len();
    }
    col_stride = _ncols;
    offset = 0;
    data.reset(_nrows*_ncols);
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < p[i].len();j++)ENTRY(i,j) = p[i][j];
    }
  }
 
  Matrix(const Matrix& M) : data(M.data), offset(M.offset), _nrows(M._nrows), _ncols(M._ncols), col_stride(M.col_stride),printall(M.printall){
    //    cout << format("Matrix copy constructor called copying %x\n",&M.data[0]);
    // NOTE: This is a soft copy, using the same dynamic storage as M
  } 

  void reset(int m, int n, const SCALAR* f = nullptr){
    data.reset(m*n);
    offset = 0;
    _nrows = m;
    col_stride = _ncols = n;
    printall = false;
    if(f != nullptr)fill(*f);
  }

  ~Matrix(void){
    //    cout << format("Matrix destructor called for %x\n",&data[0]);
  }

  bool initialized(void){return data.len() != 0;}
    
  Matrix copy(int new_cols = 0) const{ // return a new deep copy
                                       // with optional extra cols
    Matrix M(_nrows,_ncols+new_cols);
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) M(i,j) = ENTRY(i,j);
    }
    return M;
  }
  void copy(const Matrix& A){ // deep copy A to *this
    if(A._nrows != _nrows || A._ncols != _ncols)
      throw "Can't copy-- target has different dimensions\n";
    
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) ENTRY(i,j) = A(i,j);
    }
  }
  void fill(const SCALAR x){
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) ENTRY(i,j) = x;
    }
  }
  Matrix slice(int i, int j, int m, int n) const{ // shallow copy mxn submatrix at (i,j)
    assert(i >= 0 && i < _nrows && j >= 0 && j < _ncols);
    assert(m >= 0 && m <= _nrows-i && n >= 0 && n <= _ncols-j);

    Matrix M(*this);
    M.offset += i*col_stride+j;
    M._nrows = m;
    M._ncols = n;
    return M;
  }
  RowVector<SCALAR> row(int i) const { return slice(i,0,1,_ncols);}
  ColVector<SCALAR> col(int j) const { return slice(0,j,_nrows,1);}
  
  SCALAR& operator()(int i, int j) const{
#ifdef MATRIX_BOUNDS
    if(i < 0 || i >= _nrows || j < 0 || j >= _ncols){
      throw "Matrix.h: index out of bounds\n";
    }
#endif
    return data[offset + i*col_stride + j];
  }

  bool operator==(const Matrix& m) const{
    if(_nrows != m._nrows) return false;
    if(_ncols != m._ncols) return false;
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) {
        if(ENTRY(i,j) != m(i,j)) return false;
      }
    }
    return true;
  }

  bool operator!=(const Matrix& m) const{ return !operator==(m);}
  operator SCALAR(){
    if(_nrows != 1 || _ncols != 1) throw "val: Matrix size != 1x1\n";
    return ENTRY(0,0);
  }
    
  SCALAR val(void) const{
    if(_nrows != 1 || _ncols != 1) throw "val: Matrix size != 1x1\n";
    return ENTRY(0,0);
  }
  
  int nrows(void) const {return _nrows;}
  int ncols(void) const {return _ncols;}
  
  //*********

  Matrix operator+(const Matrix& A) const {
    if(_nrows != A.nrows() || _ncols != A.ncols()) throw "Matrix operator+ dimension error\n";
    Matrix B(_nrows,_ncols);
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++)B(i,j) = ENTRY(i,j)+A(i,j);
    }
    //    std::cout << "operator+ returning:\n"<<B;
    return B;
  }
  Matrix operator-(const Matrix& A) const {
    if(_nrows != A.nrows() || _ncols != A.ncols()) throw "Matrix operator- dimension error\n";
    Matrix B(_nrows,_ncols);
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++)B(i,j) = ENTRY(i,j)-A(i,j);
    }
    return B;
  }
  Matrix operator*(const Matrix& A)const {
    if(_ncols != A.nrows()) throw "Matrix multiply: dimension error\n";
    Matrix B(_nrows,A._ncols);

    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < A._ncols;j++){
        B(i,j) = 0;
        for(int k = 0;k < _ncols;k++){
          B(i,j) += ENTRY(i,k)*A(k,j);
        }
      }
    }
    return B;
  }

  Matrix operator*(double x) const{
    Matrix B(_nrows,_ncols);

    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) B(i,j) = ENTRY(i,j)*x;
    }
    return B;
  }
  
  void operator *=(SCALAR x){
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) ENTRY(i,j) *= x;
    }
  }
  void operator +=(const Matrix& A){
    if(_nrows != A.nrows() || _ncols != A.ncols()) throw "Matrix operator+= dimension error\n";
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) ENTRY(i,j) += A(i,j);
    }
  }

  void operator -=(const Matrix& A){
    if(_nrows != A.nrows() || _ncols != A.ncols()) throw "Matrix operator-= dimension error\n";
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) ENTRY(i,j) -= A(i,j);
    }
  }

  void zero(void){
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) ENTRY(i,j) = 0;
    }
  }

  // operator ColVector<SCALAR> () const { // NOTE: clang gives an error (same signature as constructor
  //   if(_ncols != 1) throw "Can't convert Matrix to ColVector.\n";
  //   return ColVector<SCALAR>(*this);
  // }

  Matrix operator*(const SparseMatrix& A); // post-multiply by sparse matrix A // NOTE: only for SCALAR = double


  Matrix operator-(void) const {
    Matrix B(_nrows,_ncols);
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++)B(i,j) = -ENTRY(i,j);
    }
    return B;
  }
  Matrix Tr(void) const {
    Matrix B(_ncols,_nrows);
    for(int i = 0;i < _ncols;i++){
      for(int j = 0;j < _nrows;j++)B(i,j) = ENTRY(j,i);
    }
    return B;
  }

  void print(void) {
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) std::cout <<ENTRY(i,j)<<" ";
      std::cout << std::endl;
    }
  }

  bool symmetric(void){
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < i;j++) {
        if(ENTRY(i,j) != ENTRY(j,i)) return false;
      }
    }
    return true;
  }

  void symmetrize(void){
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < i;j++) ENTRY(i,j) = ENTRY(j,i) = (ENTRY(i,j)+ENTRY(j,i))/2;
    }
  }
  
#ifdef DEBUG
  void dump(void) const{
    std::cout << "nrows = "<<_nrows<<", ncols = "<<_ncols<<", offset = "<<offset<<std::endl<<
      "refcount = "<< data.ref_count <<", col stride = "<<col_stride<<std::endl;
    std::cout << *this;
  }
#endif

}; // end Matrix template

template <typename SCALAR>
class RowVector : public Matrix<SCALAR>  {
  typedef Matrix<SCALAR> Base;
public:
  RowVector(void) : Base() {}
  RowVector(int n): Base(1,n) {}
  RowVector(int n, const SCALAR* s): Base(1,n,s) {}

  RowVector(std::initializer_list<SCALAR> l) {
    Base::reset(1,l.size());
    const SCALAR* p = l.begin();
    for(int j = 0;j < l.size();j++)Base::operator()(0,j) = p[j];
  };
 
  RowVector(const RowVector& M) : Base(M){
    assert(M.nrows() == 1);
  }
  
  RowVector(const Base& M) : Base(M) {
    assert(M.nrows() == 1);
  }
  void reset(int n){
    Base::reset(1,n);
  }
  int dim(void){return Base::_ncols;}
  
  SCALAR& operator[](int i){ return Base::operator()(0,i);}
  SCALAR& operator[](int i) const { return Base::operator()(0,i);}

  RowVector operator&(ColVector<SCALAR>& x) const{
    assert(x.dim() == dim());
    RowVector a(dim());
    for(int i = 0;i < dim();i++)a[i] = x[i]*(*this)[i];//(this->operator[](i)); //->operator()(0,i);
    return a;
  }

  ColVector<SCALAR> Tr(void){
    ColVector<SCALAR> a(dim());
    for(int i = 0;i < dim();i++)a[i] = (*this)[i];
    return a;
  }
};

template <typename SCALAR>
class ColVector : public Matrix<SCALAR> {
  typedef Matrix<SCALAR> Base;
public:
  ColVector(void) : Base() {}
  ColVector(int m): Base(m,1) {}
  ColVector(int m, const SCALAR* s): Base(m,1,s) {}
  

  ColVector(std::initializer_list<SCALAR> l) {
    Base::reset(l.size(),1);
    const SCALAR* p = l.begin();
    for(int i = 0;i < l.size();i++)Base::operator()(i,0) = p[i];
  };
 
  ColVector(const ColVector& M) : Base(M) {
    assert(M.ncols() == 1);
  }

  ColVector(const Matrix<SCALAR>& M) : Base(M) {
    assert(M.ncols() == 1);
  }
  
  void reset(int m, const SCALAR* fill = nullptr){
    Base::reset(m,1,fill);
  }
  int dim(void){return Base::_nrows;}

  SCALAR& operator[](int i){ return Base::operator()(i,0);}
  SCALAR& operator[](int i) const { return Base::operator()(i,0);}

  ColVector operator&(ColVector& x) const{
    assert(x.nrows() == this->_nrows);
    ColVector a(this->_nrows);
    for(int i = 0;i < this->_nrows;i++)a[i] = x[i]*(this->operator[](i)); //->operator()(i,0);
    return a;
  }
  RowVector<SCALAR> Tr(void){
    RowVector<SCALAR> a(dim());
    for(int i = 0;i < dim();i++)a[i] = (*this)[i];
    return a;
  }
  
    
};
  
template<typename SCALAR>
std::ostream& operator <<(std::ostream& os, const Matrix<SCALAR>& M){
  for(int i = 0;i < M.nrows();i++){
    for(int j = 0;j < M.ncols();j++)os << M(i,j)<<" ";
    os <<"\n";
  }
  return os;
}

template<typename SCALAR>
std::ostream& operator <<(std::ostream& os, Matrix<SCALAR>& M){
  if(M.printall){
    os << M.nrows()<<" "<<M.ncols()<<std::endl;
    M.printall = false;
  }
  for(int i = 0;i < M.nrows();i++){
    for(int j = 0;j < M.ncols();j++)os << M(i,j)<<" ";
    os <<"\n";
  }
  return os;
}

template<typename SCALAR>
std::ostream& operator <<(std::ostream& os, RowVector<SCALAR>& M){
  for(int j = 0;j < M.dim();j++)os << M[j]<<" ";
  os <<"\n";
  return os;
}

template<typename SCALAR>
std::istream& operator>>(std::istream& is, Matrix<SCALAR>& M){
  int nrows,ncols;
  is.clear();
  is >>nrows>>ncols;
  //  std::cout << "nrows: "<<nrows<<" ncols: "<<ncols<<std::endl;
  Matrix<SCALAR> M0(nrows,ncols);
  for(int i = 0;i < nrows;i++){
    for(int j = 0; j < ncols;j++) is >> M0(i,j);
  }
  //  std::cout << M0;
  M = M0;
  bool flag = is.fail();
  return is;
}

// declarations for matrix (=Matrix<double>) source in matrix.cc
matrix cholesky(const matrix& A);
bool cholesky(const matrix& M, matrix& C, double eps = 1.0e-20);
matrix sym_inv(const matrix& A, double* det = nullptr);
void symmetrize(matrix& A);
matrix inv(const matrix& A, double* det = nullptr);
bool scale_rows(matrix& A);
double dot_cols(const matrix& A, int i, int j);
double dotAB(const matrix& A, const matrix& B, int i, int j);
double gram_schmidt(matrix& A, matrix& S, double eps = 1.0e-20);
int ut0(matrix& A, double eps = 1.0e-20); // upper-triangularize in place by Givens row rotations
int ut(matrix& A, double eps = 1.0e-20); // upper-triangularize in place by Givens row rotations
matrix qr(const matrix& A);
bool reduce(matrix& A, double eps = 1.0e-20); // row-reduce upper-triangular A in place
void solve(matrix& A, double eps = 1.0e-20); // solve linear equations in-place 
double det(const matrix& A); // determinant
double trace(const matrix& A); // trace
Array<matrix> svd(const matrix& A, double eps = 1.0e-20, int maxiters = 10);
struct Svd{
  int ncols;
  int nrows;
  matrix AUV;
  matrix AU;
  matrix AV;
  matrix U;
  matrix V;
  matrix A;
  Svd(const matrix& A0){
    reduce(A0);
  }
  void reset(int nr, int nc);
  void reduce(const matrix& A0);
  Svd(void){}
  ~Svd(void){
#ifdef DEBUG
    cout << format("Svd destructor called with AUV.data = %x\n",
                   &AUV(0,0));
#endif
  }
};
Array<matrix> svd1(const matrix& A, double eps = 1.0e-20, int maxiters = 10);

class MatrixWelford { // online computation of mean and variance
  int dim;
  ColVector<double> S1;
  Matrix<double> S2;
  ColVector<double> delta;
  double W; 
  double tau; // exponential decay coefficient
public:
  MatrixWelford(void) {}
  MatrixWelford(int d, double t = 1.0);
  void reset(int d, double t = 1.0);
  void reset(void);
  void update(const ColVector<double>& x, double weight = 1.0); // add another data point
  double tot_weight(void){return W;}
  ColVector<double> mean(void){return W == 0? S1 : S1*(1.0/W);}
  Matrix<double> variance(void){return W == 0? S2: S2*(1.0/W);} 
};

#endif

