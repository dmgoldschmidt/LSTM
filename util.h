#ifndef UTIL_H
#define UTIL_H
#include<cstdarg>
#include<cstdint>
#include<string>
#include<sstream>
#include<sys/time.h>
#include<time.h>
#define MAXCHARS 1000
#include <cmath>
#include <vector>
#include <stdint.h>
#include "using.h"

#define Doub double
#define Int int

struct Erf { // from NR. This enables separate compilation (using util.cc)
	static const Int ncof=28;
	static const Doub cof[28];

	inline Doub erf(Doub x) {
		if (x >=0.) return 1.0 - erfccheb(x);
		else return erfccheb(-x) - 1.0;
	}

	inline Doub erfc(Doub x) {
		if (x >= 0.) return erfccheb(x);
		else return 2.0 - erfccheb(-x);
	}
	
  Doub erfccheb(Doub z);
  Doub inverfc(Doub p);
  inline Doub inverf(Doub p) {return inverfc(1.-p);}

};
Doub erfcc(const Doub x);

struct Normaldist : Erf {
	Doub mu, sig;
	Normaldist(Doub mmu = 0., Doub ssig = 1.) : mu(mmu), sig(ssig) {
		if (sig <= 0.) throw("bad sig in Normaldist");
	}
	Doub p(Doub x) {
      double v = (x-mu)/sig;
      return (0.398942280401432678/sig)*exp(-0.5*v*v);
	}
	Doub cdf(Doub x) {
		return 0.5*erfc(-0.707106781186547524*(x-mu)/sig);
	}
	Doub invcdf(Doub p) {
		if (p <= 0. || p >= 1.) throw("bad p in Normaldist");
		return -1.41421356237309505*sig*inverfc(2.*p)+mu;
	}
};

struct Lognormaldist : Erf {
	Doub mu, sig;
	Lognormaldist(Doub mmu = 0., Doub ssig = 1.) : mu(mmu), sig(ssig) {
		if (sig <= 0.) throw("bad sig in Lognormaldist");
	}
	Doub p(Doub x) {
		if (x < 0.) throw("bad x in Lognormaldist");
		if (x == 0.) return 0.;
		return (0.398942280401432678/(sig*x))*exp(-0.5*sqrt((log(x)-mu)/sig));
	}
	Doub cdf(Doub x) {
		if (x < 0.) throw("bad x in Lognormaldist");
		if (x == 0.) return 0.;
		return 0.5*erfc(-0.707106781186547524*(log(x)-mu)/sig);
	}
	Doub invcdf(Doub p) {
		if (p <= 0. || p >= 1.) throw("bad p in Lognormaldist");
		return exp(-1.41421356237309505*sig*inverfc(2.*p)+mu);
	}
};

class LFSR128 { // implements the xorshift128+ algorithm
  uint64_t s[2];
public:
  LFSR128(void) {}
  LFSR128(uint64_t seed){ 
    step(seed);
    for(int i = 0;i < 1000;i++) step();
  }
  LFSR128(uint64_t* ss) {
    step(ss);
    for(int i = 0;i < 1000;i++) step();
  }
  uint64_t step(void){
	uint64_t x = s[0];
	uint64_t const y = s[1];
	s[0] = y;
	x ^= x << 23; // a
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
	return s[1] + y;
  }
  uint64_t step(uint64_t seed){
    s[0] = seed;
    s[1] = 0;
    return step();
  }
  uint64_t step(uint64_t* ss){
    s[0] = ss[0];
    s[1] = ss[1];
    return step();
  }
  double uniform(void){ return step()/(double)UINT64_MAX;}
  LFSR128& operator=(LFSR128& x){
    s[0] = x.s[0];
    s[1] = x.s[1];
    return *this;
  }
};

struct Normaldev : LFSR128 { // from NR
  Doub mu,sig;
  Normaldev(void) {}
  Normaldev(Doub mmu, Doub ssig, uint64_t i)
	: LFSR128(i), mu(mmu), sig(ssig){}
  Doub dev(void);
  void reset(Doub mmu, Doub ssig){ 
    mu = mmu;
    sig = ssig;
  }
  void reset(uint64_t i){ // re-seed
    step(i);
  }
};

class PopCount {
  char byte_count[256];
 public:
  PopCount(void);
  uint operator()(uint64_t word);
};

class LCG64 { // 64-bit LCG. Parameters from Wikipedia
  static const uint64_t a;
  static const uint64_t b;
  static const double uint64_max;
  uint64_t x;
public:
  LCG64(uint64_t seed = 0){// : a(2862933555777941757), b(3037000493), uint64_max((double)UINT64_MAX){
    reseed(seed);
  }
  LCG64(const LCG64& lcg){// : a(2862933555777941757), b(3037000493), uint64_max((double)UINT64_MAX){
    x = lcg.x; // copy state
  }
  uint64_t step(void){return x = a*x+b;}
  double uniform(void){ return step()/uint64_max;}
  uint64_t reseed(uint64_t seed){
    x = seed;
    for(int i = 0;i < 10;i++)step(); // avoid crappy seeds
    return step();
  }
  LCG64& operator=(const LCG64& lcg){
    x = lcg.x;
    return *this;
  }
};

class BloomFilter { // Have we seen a 64-bit key or a string yet? Remember it for next time
  uint64_t word[4];
  LCG64 lcg;
  int _density;
 public:
  BloomFilter(void) { clear();}
  bool hash(uint64_t key, bool set = true){
    bool ret = true;
    uint64_t hash_val = lcg.reseed(key);
    for(int i = 0;i < 8;i++){
      int j = hash_val & 3; // low order 2-bits
      hash_val >>= 2;
      uint64_t mask  = (1<<(hash_val & (0x3f))); // next six bits
      hash_val >>= 6;
      if(  (mask&word[j]) == 0){ // bit not set
        ret = false; 
        if(set){ // set the bit
          word[j] |= mask; 
          _density++;
        }
      }
    }
    return ret;
  }
  bool hash(std::string& s, bool set = true){
    uint64_t hval = 0;
    for(int i = 0;i < s.size();i++){
      hval ^= (s[i] << (i%8));
      return hash(hval,set);
    }
  }
  void clear(void){
    for(int i = 0;i < 4;i++){
      word[i] = 0;
      _density = 0;
    }
  }
  bool has(uint64_t key){return hash(key,false);}
  bool has(std::string& s){return hash(s,false);}
  int density(void){ return _density; }
};
      
      


#define MAXFIELDS 100  //Not easy to make field an Array because the header references are circular

class StringSplitter { // Functor to split a C-string into fields using field separator fs.  NOTE: leading
  // blanks in a field are skipped, so the field separator  is really fs followed by any number of blanks.
  // If fs == ' ', the splitter will also recognize '\t'.  If fs == '\t', ONLY '\t' is recognized.
  int nf; // no. of fields split out
  char* field[MAXFIELDS]; // array of ptrs to fields
public:
  StringSplitter(void):  nf(0) {} 
  int operator()(char* record,char fs = ' '); /* input a record, split into fields using field separator fs, 
                                               * return nf (Note: fs is over-written with '\0', making
                                               * record a sequence of C-strings)*/
  char* operator[](int i); // subsequent access to the fields

};


void split_string(char* line, std::vector<char*>& token, char fs = ' ');
void deblank(char* string);
std::string format(const char* fstr,...);
std::string str_time(double t);
std::string str_date(double t);
double get_time(int date, std::string time);
std::string currentDateTime(void);
class InternalTime { // floating pt seconds <-> 38-bit integer microseconds (3.18 days = 2^38 microseconds)
  int a;
  char b;
  inline void convert(double delta){
    double n = delta*1000000/128;
    a = (int) n; // high-order signed 31 bits
    b = (char) ((n-a)*128); // low-order signed 7 bits
  }
public:
  InternalTime(double delta = 0){ 
    if(fabs(delta) > 259200) throw format("InternalTime: bad input %lf\n",delta); // abs(delta) >= 3 days
    convert(delta);
  }
  inline InternalTime& operator=(double delta){convert(delta); return *this;}
  inline InternalTime& operator=(const InternalTime& t){a = t.a;b = t.b; return *this;}
  inline operator double() const{return (a*128.0 + b)/1000000.0;}
};

uint32_t str_hash(const char* s, uintptr_t salt = 0x55555555); // hash a c-string

class String : public std::string { // example of a useable KEY class (see KeyIndex.h) 
public:
  static String null;
  String(void) : std::string() {}
  String(const std::string& s) : std::string(s){}
  String(std::string& s) : std::string(s){}
  String(const char* p) : std::string(p){}
  String(char* p) : std::string(p){}
  uint32_t hash(uintptr_t salt = 0) const{return str_hash(this->c_str(),salt);}
  uint32_t hash1(uintptr_t salt = 0) const; // alt hash function
  operator int() const{ return hash();}
  template<typename T>
  String(T x){ // also add type conversions from int, double, char, and other classes 
    std::stringstream iss;
    iss << x;
    *this = iss.str();
  }
};

class Welford { // online computation of mean and variance
  double S1;
  double S2;
  double W;
  double tau; // exponential decay coefficient
  std::string ser;
public:
  Welford(double t = 1.0) : W(0),S1(0),S2(0),tau(t) {}
  void reset(double t = 1.0){W = S1 = S2 = 0;tau = t;}
  std::string& serialize(void){
    ser = format("%.18f %.18f %.18f %.18f",S1,S2,W,tau); // for PersistentState
    return ser;
  }
  Welford(std::string& s){ restart(s);}
  void restart(std::string& s){ // restart from serialization
    std::istringstream iss(s);
    iss >> S1 >> S2 >> W >> tau;
  }
  void update(double x, double weight = 1.0); // add another data point
  double tot_weight(void){return W;}
  double mean(void){return W == 0? 0:S1/W;}
  double variance(void){return W == 0? 0 : S2/W;} // NOTE: for correctness, should be n-1, but WGAS.
};

#endif
