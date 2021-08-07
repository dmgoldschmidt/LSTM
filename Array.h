/* This is a storage container that solves two common problems with arrays:
 * 1. You can't pass them to or from a function efficiently.  The old C solution
 * is to pass pointers, but this doesn't let the function construct an array and
 * return it to the caller.  So the caller has to construct the array (often without 
 * knowing how big it should be) and give the function a pointer to it.

* 2. You sometimes don't know how big the array should be in advance.  This problem is
 * "solved" by the STL vector class, but the syntax using the push_back method is clumsy.

 * The Array class solves both problems.  
 * Solution to 1: An Array object just consists of a single pointer, so
 * it can be passed around and copied efficiently. The pointer points to an internal ArrayBlock
 * object on the heap, which among other things, keeps a reference count of how many Array objects
 * are pointing to it. The refcount is updated by all assignment, copy-constructor, and destruction
 * operations, and if and when it goes to zero the heap storage is released. 
 * For example, the following code looks really bad:
 *    Array<double> foo(void){
        Array<double> A(100000);
        .......
        return A;
      }
 * but it's actually fine.  The Array does not get copied by the return statement, nor does it
 * get deleted when it appears to go out of scope, nor does it cause a memory leak by hanging
 * around when it's no longer in use.  

 * Solution to 2: Arrays are variable length.  If a reference is made to an Array location higher 
 * than the currently allocated length, additional storage is allocated to accommodate that reference.
 * To catch bugs and prevent enormous allocations,  the user can set a maximum length using the set_max_length
 * method. This does not actually allocate any storage, it just sets a limit which generates an exception
 * if the user tries to reference a location beyond the limit.
 * The storage is organized as a linked list of fixed-length blocks, where the block-length is specified by the
 * user in the constructor.  Example:

 * Array<int> A(10);  // An initial block of 10 ints is allocated
 * A[25] = 2; // two more blocks are added to the linked list to accommodate the reference
 * A.set_max_length(100); 
 * A[101] = 6; // exception thrown!

 * So we can think of an Array as interpolating between an ordinary array and a linked list, with the block size
 * being the interpolation parameter.  A small parameter is space-efficient but time-inefficient, while a large
 * parameter is space-inefficient but time efficient.

 */
#ifndef ARRAY_H
#define ARRAY_H
#include <iostream>
#include <cassert>
//#include <vector>
#include "using.h"
#include "util.h"
//#define DEBUG
#undef DEBUG
#ifdef DEBUG
#define DBG(x) x
#else
#define DBG(x)
#endif


struct ArrayBoundsError {
  int bound;
  int index;
  ArrayBoundsError(int b, int i) : bound(b), index(i){}
};

template <typename ITEM>
class Array; 

template <typename ITEM>
struct ArrayInitializer { // subclass this for a callback initialization functor
  virtual void operator()(ITEM& item) = 0; // if specified, it is called once for each allocated ITEM
};

template <typename ITEM>
struct RowInit : public ArrayInitializer<Array<ITEM>> {
  /*  this is used for matrices with a variable no. of rows, like Array<Array<ITEM>> example
   * so example calls operator() for each new Array<ITEM>& row and the row gets reset to the desired length with *fill
   * if this isn't specified when example is constructed or reset and you reference a row beyond the initial
   * allocation without resetting it yourself, you will get an ArrayBoundsError.

   * example of usage:
   * int zero(0);
   * RowInit<int> my_row_init(10,&zero);
   * Array<Array<int>> example(100,&my_row_init);
   * example now has 100 zero rows each of length 10
   * example[101][0] = 1;
   * example now has 200 rows which are all zero except for example[101][0] = 1;
   */
  int ncols;
  ITEM* fill;
  RowInit(void){}
  RowInit(int n, ITEM *f = nullptr) : ncols(n), fill(f) {} // length of the row and initial contents
  void operator()(Array<ITEM>& row){row.reset(ncols,fill);}
  void reset(int n, ITEM *f = nullptr){
    ncols = n;
    fill = f;
  }
};

template <typename ITEM>
class ArrayBlock{ // this is a private helper class for Array
  friend class Array<ITEM>; 
  size_t n; // block size
  ITEM *data; // block of n items
  ArrayBlock* next; // pointer to next block
  uint refcount; // only used in the first block
  int max_index; // ditto. If non-zero, bounds the total length
  int highest_index; // ditto. Tracks the highest array loc. referenced so far
  ArrayInitializer<ITEM>* init; // user defined initialization functor
  const ITEM* fill; // alternative initialization to constant

  // ArrayBlock(size_t nn, const ITEM* f) : n(nn), next(nullptr), refcount(0), fill(f) {

  //   data = new ITEM[n];
  //   DBG(std::cout << format("new ArrayBlock at %x, data at %x\n",this,data);)
  //   if(fill != nullptr)
  //     for(uint i = 0;i < n;i++)data[i] = *fill;
  // }
 ArrayBlock(size_t nn, ArrayInitializer<ITEM>* in)
   : n(nn), next(nullptr), refcount(0), max_index(-1), highest_index(-1), init(in), fill(nullptr) {
    // this constructor uses a user-defined initializer
   data = new ITEM[n];
   DBG(std::cout << format("new ArrayBlock at %x, data at %x\n",this,data);)
   if(init != nullptr){
     for(uint i = 0;i < n;i++){
       (*init)(data[i]); // call user-defined functor to initialize each ITEM
     }
   }
 }

 ArrayBlock(size_t nn, const ITEM* item) : n(nn), next(nullptr), refcount(0),
    max_index(-1), highest_index(-1),  init(nullptr), fill(item) {
   
   // this constructor initializes to a user-defined constant
    data = new ITEM[n];
    DBG(std::cout << format("new ArrayBlock object at %x, data at %x\n",this,data);)
    if(fill != nullptr){
      for(uint i = 0;i < n;i++)data[i] = *fill; // initialize with constant
      highest_index = n-1;
    }
 }
  ArrayBlock(const ArrayBlock&); // no copy
  ArrayBlock& operator=(const ArrayBlock&); //no assignment

  ~ArrayBlock(void){
    DBG(std::cout << format("~ArrayBlock: deleting object at  %x and data at %x\n",this,data);)
    delete[] data;
    if(next != nullptr)delete next;
  }

  //  uint len(void){ // recursive 
  //    return n + (next == nullptr? 0: next->len());
  //  }


  ITEM& operator[](uint i){ // recursive
    DBG(std::cout << format("ArrayBlock %x: operator [%d]\n",this,i);)
      if(n <= 0) throw ArrayBoundsError(0,i);  
    if(i >= n){ // on to the next block
      DBG(std::cout << format("index %d exceeds block length\n",i);)
	//	exit(1);
	//      fflush(stdout);
      if(next == nullptr){
        if(init != nullptr) next = new ArrayBlock(n,init);
        else next = new ArrayBlock(n,fill);
      }
      return next->operator[](i-n);
      
    }
    return data[i];
  }
};
// Main code starts here

template<typename ITEM> 
class Array {
  //  typedef void(*ArrayInitializer)(ITEM&);
  //  friend ostream& operator <<(ostream& os, const Array& A);
protected:
  ArrayBlock<ITEM>* first; // first (and possibly only) block of data and control info
public:
  Array(void) : first(nullptr){
    DBG(std::cout << format("Array %x void constructor\n",this);)
  }
  // Array(size_t n, const ITEM* fill = nullptr) : first(new ArrayBlock<ITEM>(n,fill)){
  //   first->refcount = 1;
  //   DBG(std::cout << format("Array %x: first block at %x\n",this,first);)
  // }
 Array(size_t n, ArrayInitializer<ITEM>* init) : first(new ArrayBlock<ITEM>(n,init)){ // user supplies callback routine
                                                                                 // to initialize each ITEM (see above)      
    if(n == 0) throw "Array: Can't initialize to length zero.\n";
    first->refcount = 1;
    DBG(std::cout << format("Array %x: first block at %x\n",this,first);)
  }
  Array(size_t n, const ITEM* fill = nullptr) : first(new ArrayBlock<ITEM>(n,fill)){
    if(n == 0) throw "Array: Can't initialize to length zero.\n";
    first->refcount = 1;
    DBG(std::cout << format("Array %x: first block at %x\n",this,first);)
  }
  Array(std::initializer_list<ITEM> l){ // note: the allocation block size is the length of the list
    int n = l.size();
    first = new ArrayBlock<ITEM>(n,(ITEM*)NULL);
    first->refcount = 1;
    const ITEM* p = l.begin();
    for(int i = 0;i < l.size();i++)(*first)[i] = p[i];
    first-> highest_index = l.size()-1;
  }

  Array(const Array& A): first(A.first) { // shallow copy
    if(first != nullptr) first->refcount++; 
    DBG(if(first != nullptr) std::cout << format("Array %x (Block %x) copy constr: refcount = %d\n",this,first,first->refcount);) 
   }
  ~Array(void){
    DBG(if(first != nullptr) std::cout<< format("~Array %x (Block %x): refcount = %d\n",this,first,first->refcount);) 
    if( first != nullptr && first->refcount-- <= 1)delete first;
  }
  void reset(size_t n, ArrayInitializer<ITEM>* init){
    DBG(if(first != nullptr) std::cout << format("Array %x(Block %x) reset: refcount = %d\n",this,first,first->refcount);) 
    if(first != nullptr && first->refcount-- <= 1)delete first;
    first = new ArrayBlock<ITEM>(n,init);
    first->refcount = 1;
  }

  void reset(size_t n, const ITEM* fill = nullptr){
    DBG(if(first != nullptr) std::cout << format("Array %x(Block %x) reset: refcount = %d\n",this,first,first->refcount);) 
    if(first != nullptr && first->refcount-- <= 1)delete first;
    first = new ArrayBlock<ITEM>(n,fill);
    first->refcount = 1;
    first->highest_index = n-1;
  }

  Array& operator=(const Array& A){ // shallow copy -- switch pointer to new first block
    DBG(if(first != nullptr) std::cout << format("Array %x(Block %x) assignment: refcount = %d\n",this,first,first->refcount);)
    if(first != nullptr && first->refcount-- <= 1)delete first;
    first = A.first;
    if(first != nullptr)first-> refcount++;
    DBG(if(first != nullptr) std::cout << format("Array %x(Block %x) assignment: refcount = %d\n",this,first,first->refcount);)
    return *this;
  }
  Array copy(void) const{ // deep copy returning object 
    Array<ITEM> A;
    if(first != nullptr){
      if(first->init != nullptr)A.reset(first->n,first->init); 
      else A.reset(first->n,first->fill);
      A.first->highest_index = first->highest_index;
      A.first->max_index = first->max_index;
      int n = first->highest_index; 
      for(int i = 0;i <= n;i++) 
        A[i] = this->operator[](i); // copy the data.  This automagically allocates any 
                                                           // needed extra blocks
    }
    return A;
  }
  void fill(ITEM x){ // fill all currently allocated blocks
    for(auto current = first;current != nullptr;current = current->next){
      for(int i = 0;i < current->n;i++)first->operator[](i) = x;
    }
  }

  bool set_max_length(int m){ // max length = 0 means we're not bounding the length
    if(first == nullptr ) return false; // no can do
    first->max_index = first->highest_index = m-1;
    return true;
  }
  int max_length(void) const {
    if(first == nullptr) return 0;
    return first->max_index+1;
  }
  int len(void) const {
    if(first == nullptr) return 0;
    return first->highest_index+1;
  }
  void zero(void){ fill(0);}

  size_t blocksize(void){
    if(first != nullptr)return first->n;
    return 0;
  }
  
  ITEM& operator[](int i)const {
    if(first == nullptr || i < 0)throw ArrayBoundsError(-1,i);
    if(first->max_index > -1 && i > first->max_index) throw ArrayBoundsError(first->max_index,i);
    if(i > first->highest_index) first->highest_index = i;
    return first->operator[](i);
  }
  bool operator==(Array& A) const{
    if(A.max_length() != max_length())return false;
    int n = len();
    if(n != A.len())return false;
    for(int i = 0;i < n;i++) if(this->operator[](i) != A[i])return false;
    return true;
  }
  bool operator!=(Array& A) const{return !operator==(A);}

  const std::string str(void) const{
    std::ostringstream oss;
    for(int i = 0;i < len();i++)oss << operator[](i)<<" ";
    return oss.str();
  }
};


template<typename ITEM>
std::ostream& operator <<(std::ostream& os, Array<ITEM>& A){
  int n = A.len();
  for(int i = 0;i < n;i++)os << A[i]<<" ";
  return os;
} 
template<typename ITEM>
std::ostream& operator <<(std::ostream& os, const Array<ITEM>& A){
  int n = A.len();
  for(int i = 0;i < n;i++)os << A[i]<<" ";
  return os;
} 

template<typename ITEM>
std::istream& operator>>(std::istream& is, Array<ITEM>& A){
  int len,blocksize;
  is >> len >> blocksize;
  Array<ITEM> A0(blocksize);
  for(int i = 0;i < len;i++) is >> A0[i];
  A = A0;
  return is;
}
  

#endif
/*#define REFCOUNT (((int*)buf)[0])
#define N (((int*)buf)[1])
#define DATA ((ITEM*)(buf+Hlen*sizeof(int)/sizeof(ITEM) + 1)) // Note: is this right?
*/
  
  // void reset(const ITEM* p){first->reset(p);}
  // void reset(const ITEM& fill){first->reset(fill);}
  // void reset(void){first->reset();}
  //   if(first != nullptr){
  //     if(n == N) return; //NOP
  //     if(--REFCOUNT <= 0)delete[] buf;
  //   }
  //   int nitems = n + Hlen*sizeof(int)/sizeof(ITEM) + 1;
  //   buf = new ITEM[nitems];
  //   DBG << format("allocated Array of %d items of length %d at %x\n",nitems,sizeof(ITEM),buf); 
  //   REFCOUNT = 1;
  //   N = n;
  // }

// template <typename ITEM, size_t N>
// class ArrayBlock{ // linked list of data blocks and control info
//   friend class Array<ITEM,N>;
//   ITEM data[N];
//   //  uint start;
//   ArrayBlock* next;
//   uint refcount;
//   ITEM fill;
//   bool filling;

//   ArrayBlock(void) : next(nullptr), refcount(0), filling(false) {}
//   ArrayBlock(const ITEM& f) : fill(f), next(nullptr), refcount(0), filling(true) {
//     for(uint i = 0;i < N;i++)data[i] = fill;
//   }
//   ArrayBlock(const ArrayBlock&); // no copy
//   ArrayBlock& operator=(const ArrayBlock&); //no assignment

//   ~ArrayBlock(void){
//     if(next != nullptr)delete next;
//   }

//   uint len(void){
//     return N + (next == nullptr? 0: next->len());
//   }

//   ITEM& operator[](uint i){
//     DBG(format("ArrayBlock %x: operator [%d]\n",this,i))
//     if(i >= N){ // on to next block
//       DBG(format("index %d exceeds block length\n",i))
//       fflush(stdout);
//       if(next == nullptr){
// 	next = filling? new ArrayBlock(fill): new ArrayBlock();
// 	DBG(format("new ArrayBlock at %x\n",next))
// 	fflush(stdout);
//       }
//       return next->operator[](i-N);
      
//     }
//     return data[i];
//   }
// };
