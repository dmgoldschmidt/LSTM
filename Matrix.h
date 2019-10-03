#ifndef MATRIX_H
#define MATRIX_H
#include "Array.h"
#include "util.h"

// #define COL_STRIDE (((int*)Base::buf)[2])
// #define MDATA ((SCALAR*)(Base::buf+3*sizeof(int)/sizeof(SCALAR) + 1))

#define ENTRY (operator())

template<typename SCALAR>
class Matrix;
typedef Matrix<double> matrix;
typedef Matrix<int> imatrix;
class SparseMatrix;
template<typename SCALAR>
class ColVector;

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
    //    cout << "sin_t = "<<sin_t<<", cos_t = "<<cos_t<<endl;
    return true;
  }
   
  inline void rotate(double& x, double& y){
    double z = x;
    x = cos_t*z + sin_t*y;
    y = -sin_t*z + cos_t*y;
  }
};

template<typename SCALAR>
class Matrix : public Array<SCALAR> {
protected:
  typedef Array<SCALAR> Base;
  int offset; // these parameters support strides
  int _nrows;
  int _ncols;
  int col_stride;
public:
  bool printall;
  Matrix(void) : Base(), offset(0), _nrows(0), _ncols(0), col_stride(0), printall(false) {}
  Matrix(int m, int n, const SCALAR* fill = nullptr ): Base(m*n,fill), offset(0), _nrows(m), _ncols(n), col_stride(n),
                                                       printall(false){}

  Matrix(initializer_list<Array<SCALAR>> l) : printall(false){
    
    _nrows = l.size();
    _ncols = 0;
    const Array<SCALAR>* p = l.begin();
    
    for(int i = 0;i < _nrows;i++){
      if(p[i].len() > _ncols)_ncols = p[i].len();
    }
    col_stride = _ncols;
    offset = 0;
    Base::reset(_nrows*_ncols);
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < p[i].len();j++)ENTRY(i,j) = p[i][j];
    }
  };
 
  Matrix(const Matrix& M) : Base(M), offset(M.offset), _nrows(M._nrows), _ncols(M._ncols), col_stride(M.col_stride),
                            printall(M.printall){} // uses Array shallow copy
  
  void reset(int m, int n, const SCALAR* fill = nullptr){
    Base::reset(m*n,fill);
    offset = 0;
    _nrows = m;
    col_stride = _ncols = n;
    printall = false;
  }
  Matrix copy(int new_cols = 0) const{ // return a new deep copy
                                       // with optional extra cols
    Matrix M(_nrows,_ncols+new_cols);
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++) M(i,j) = ENTRY(i,j);
    }
    return M;
  }
  void copy(const Matrix& A){ // deep copy A to *this
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
    Matrix M(*this);
    M.offset = i*col_stride+j;
    M._nrows = m;
    M._ncols = n;
    return M;
  }
    
  SCALAR& operator()(int i, int j) const{
#ifdef MATRIX_BOUNDS
    if(i < 0 || i >= _nrows || j < 0 || j >= _ncols){
      throw "Matrix.h: index out of bounds\n";
    }
#endif
    return Base::operator[](offset + i*col_stride + j);
  }

  bool operator==(const Matrix& m) const{
    if(offset != m.offset) return false;
    if(_nrows != m._nrows) return false;
    if(_ncols != m._ncols) return false;
    if(col_stride != m.col_stride) return false;
    Array<SCALAR> m0 = (Array<SCALAR>)*this; // this ridiculous code works around an apparent g++ bug
    Array<SCALAR> m1 = (Array<SCALAR>)m;
    return m0 == m1;
       //    return (Array<SCALAR>)(*this) == (Array<SCALAR>)m;
  }

  bool operator!=(const Matrix& m) const{ return !operator==(m);}


  int nrows(void) const {return _nrows;}
  int ncols(void) const {return _ncols;}
  //*********

  Matrix operator+(const Matrix& A) const {
    if(_nrows != A.nrows() || _ncols != A.ncols()) throw "Matrix operator+ dimension error\n";
    Matrix B(_nrows,_ncols);
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++)B(i,j) = ENTRY(i,j)+A(i,j);
    }
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
    if(_ncols != A.nrows()) throw "Matrix operator* dimension error\n";
    Matrix B(_nrows,A._ncols);

    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < A._ncols;j++){
        B(i,j) = 0;
        for(int k = 0;k < _ncols;k++){
          B(i,j) += ENTRY(i,k)*A(k,j);
          //          cout << format("ENTRY(%d,%d) = %d, A(%d,%d) = %d, B(%d,%d) = %d\n",i,k,ENTRY(i,k),k,j,A(k,j),i,j,B(i,j));
        }
      }
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

  operator ColVector<SCALAR> () const { return ColVector<SCALAR>(*this);}

  Matrix operator*(const SparseMatrix& A); // post-multiply by sparse matrix A // NOTE: only for SCALAR = double


  Matrix operator-(void) const {
    Matrix B(_nrows,_ncols);
    for(int i = 0;i < _nrows;i++){
      for(int j = 0;j < _ncols;j++)B(i,j) = -ENTRY(i,j);
    }
    return B;
  }
  Matrix T(void) const {
    Matrix B(_ncols,_nrows);
    for(int i = 0;i < _ncols;i++){
      for(int j = 0;j < _nrows;j++)B(i,j) = ENTRY(j,i);
    }
    return B;
  }
        
#ifdef DEBUG
  void dump(void) const{
    std::cout << "nrows = "<<_nrows<<", ncols = "<<_ncols<<", offset = "<<offset<<std::endl<<
      "refcount = "<< Base::first->refcount <<", col stride = "<<col_stride<<std::endl;
    std::cout << *this;
  }
#endif

}; // end Matrix template

template <typename SCALAR>
class RowVector : public Matrix<SCALAR>  {
  typedef Matrix<SCALAR> Base;
public:
  RowVector(void) : Base() {}
  RowVector(int n, const SCALAR* fill = nullptr ): Base(1,n,fill) {}

  RowVector(initializer_list<SCALAR> l) {
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
  void reset(int n, const SCALAR* fill = nullptr){
    Base::reset(1,n,fill);
  }
};

template <typename SCALAR>
class ColVector : public Matrix<SCALAR> {
  typedef Matrix<SCALAR> Base;
public:
  ColVector(void) : Base() {}
  ColVector(int m, const SCALAR* fill = nullptr ): Base(m,1,fill) {}

  ColVector(initializer_list<SCALAR> l) {
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

  ColVector operator&(const ColVector& x){
    assert(x.nrows() == this->_nrows);
    ColVector<double> a(this->_nrows);
    for(int i = 0;i < this->_nrows;i++)a[i] = x[i]*this->operator()(i,0);
    return a;
  }
    
};
  
class SparseMatrix {
  int _nrows;
  int _ncols;
  int _nentries; // no. of tabulated entries = no. of rows of A 
  int transposed;
public:
  matrix Entries; // A(m,0) = row index i, A(m,1) = col index j, A(m,2) = (i,j)-entry 
  SparseMatrix(void){}
  SparseMatrix(int m, int n, const matrix& A);
  void reset(int m, int n, const matrix& A);
  matrix operator*(const matrix& M); // premultiply dense matrix M by sparse matrix *this
  matrix svd1(int& r, int niters2 = 100, double eps = 1.0e-8, unsigned seed = 1);
  matrix svd(int r, int niters = 100, double eps = 1.0e-8, unsigned seed = 1);
  int row(int m) const;
  int col(int m) const;
  double entry(int m) const;
   
  int nrows(void) const;
  int ncols(void) const;
  int nentries(void) const;
  void transpose(void){transposed ^= 1;} // flip transposed flag
  void set_transpose_flag(bool b){transposed = (int)b;}
  void swap_rows(int i1, int i2);
  void reheap(int i , int n);
  int compare(int i, int j, int n);
  Array<int> sort(void); // sort entries in place by row index, return row starts
  matrix abs_mult(const matrix& M); // premultiply M by *this, ignoring signs
};

template<typename SCALAR>
Matrix<SCALAR> Matrix<SCALAR>::operator*(const SparseMatrix& A){ // post-multiply by sparse matrix A // NOTE: only for SCALAR = double
  double zero = 0;
  if(_ncols != A.nrows()) throw "Sparse post-multiply dimension error\n";
  Matrix B(_nrows,A.ncols(),&zero);
  for(int m = 0;m < A.nentries();m++){
    int j = A.row(m);
    int k = A.col(m);
    for(int i = 0;i < _nrows;i++){ 
      B(i,k) += ENTRY(i,j)*A.entry(m);
      //	cout << format("B(%d,%d) = %f\n",i,k,B(i,k));
    }
  }
  return B;
}


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
    os << M.nrows()<<" "<<M.ncols()<<endl;
    M.printall = false;
  }
  for(int i = 0;i < M.nrows();i++){
    for(int j = 0;j < M.ncols();j++)os << M(i,j)<<" ";
    os <<"\n";
  }
  return os;
}

template<typename SCALAR>
std::istream& operator>>(std::istream& is, Matrix<SCALAR>& M){
  int nrows,ncols;
  is.clear();
  is >>nrows>>ncols;
  //  cout << "nrows: "<<nrows<<" ncols: "<<ncols<<endl;
  Matrix<SCALAR> M0(nrows,ncols);
  for(int i = 0;i < nrows;i++){
    for(int j = 0; j < ncols;j++) is >> M0(i,j);
  }
  //  cout << M0;
  M = M0;
  bool flag = is.fail();
  return is;
}

// declarations for matrix (=Matrix<double>)
bool scale_rows(matrix& A);
double dot_cols(const matrix& A, int i, int j);
double dotAB(const matrix& A, const matrix& B, int i, int j);
double gram_schmidt(matrix& A, matrix& S, double eps = 1.0e-8);
int ut0(matrix& A, double eps = 1.0e-10); // upper-triangularize in place by Givens row rotations
int ut(matrix& A, double eps = 1.0e-10); // upper-triangularize in place by Givens row rotations
matrix qr(const matrix& A);
void reduce(matrix& A, double eps = 1.0e-10); // row-reduce upper-triangular A in place
void solve(matrix& A, double eps = 1.0e-10); // solve linear equations in-place 
matrix inv(const matrix& A, double eps = 1.0e-10);
Array<matrix> svd(const matrix& A, double eps = 1.0e-10, int maxiters = 10);
struct Svd{
  int ncols;
  int nrows;
  matrix AUV;
  matrix AU;
  matrix AV;
  matrix U;
  matrix V;
  matrix A;

  void reduce(const matrix& A0);
};
Array<matrix> svd1(const matrix& A, double eps = 1.0e-10, int maxiters = 10);

#endif
