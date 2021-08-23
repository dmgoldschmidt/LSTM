#include "util.h"
#include "stats.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cctype>

using namespace std;


#define Doub double
#define Int int
double eps = std::numeric_limits<Doub>::epsilon();
double fpmin = std::numeric_limits<Doub>::min()/eps;

const Doub Erf::cof[28] = {-1.3026537197817094, 6.4196979235649026e-1,
                           1.9476473204185836e-2,-9.561514786808631e-3,-9.46595344482036e-4,
                           3.66839497852761e-4,4.2523324806907e-5,-2.0278578112534e-5,
                           -1.624290004647e-6,1.303655835580e-6,1.5626441722e-8,-8.5238095915e-8,
                           6.529054439e-9,5.059343495e-9,-9.91364156e-10,-2.27365122e-10,
                           9.6467911e-11, 2.394038e-12,-6.886027e-12,8.94487e-13, 3.13092e-13,
                           -1.12708e-13,3.81e-16,7.106e-15,-1.523e-15,-9.4e-17,1.21e-16,-2.8e-17};
	
Doub Erf::erfccheb(Doub z){
  Int j;
  Doub t,ty,tmp,d=0.,dd=0.;
  if (z < 0.) throw("erfccheb requires nonnegative argument");
  t = 2./(2.+z);
  ty = 4.*t - 2.;
  for (j=ncof-1;j>0;j--) {
    tmp = d;
    d = ty*d - dd + cof[j];
    dd = tmp;
  }
  return t*exp(-z*z + 0.5*(cof[0] + ty*d) - dd);
}

Doub gammln(const Doub xx){
  Int j;
  Doub x,tmp,y,ser;
  static const Doub cof[14]={57.1562356658629235,-59.5979603554754912,
                             14.1360979747417471,-0.491913816097620199,.339946499848118887e-4,
                             .465236289270485756e-4,-.983744753048795646e-4,.158088703224912494e-3,
                             -.210264441724104883e-3,.217439618115212643e-3,-.164318106536763890e-3,
                             .844182239838527433e-4,-.261908384015814087e-4,.368991826595316234e-5};
  if (xx <= 0) throw("bad arg in gammln");
  y=x=xx;
  tmp = x+5.24218750000000000;
  tmp = (x+0.5)*log(tmp)-tmp;
  ser = 0.999999999999997092;
  for (j=0;j<14;j++) ser += cof[j]/++y;
  return tmp+log(2.5066282746310005*ser/x);
}
Doub gser(const Doub a, const Doub x) {
  Doub sum,del,ap;
  Doub gln=gammln(a);
  ap=a;
  del=sum=1.0/a;
  for (;;) {
    ++ap;
    del *= x/ap;
    sum += del;
    if (fabs(del) < fabs(sum)*eps) {
      return sum*exp(-x+a*log(x)-gln);
    }
  }
}
Doub gcf(const Doub a, const Doub x) {
  Int i;
  Doub an,b,c,d,del,h;
  Doub gln=gammln(a);
  b=x+1.0-a;
  c=1.0/fpmin;
  d=1.0/b;
  h=d;
  for (i=1;;i++) {
    an = -i*(i-a);
    b += 2.0;
    d=an*d+b;
    if (fabs(d) < fpmin) d=fpmin;
    c=b+an/c;
    if (fabs(c) < fpmin) c=fpmin;
    d=1.0/d;
    del=d*c;
    h *= del;
    if (fabs(del-1.0) <= eps) break;
  }
  return exp(-x+a*log(x)-gln)*h;
}
Doub gammq(const Doub a, const Doub x) { // incomplete gamma function (chi-square distr.)
  if (x < 0.0 || a <= 0.0 || a > 100) throw("bad args in gammq"); 
  // NOTE:  NR uses Gaussian quadrature for a > 100 ( a = deg. of freedom)
  if (x == 0.0) return 1.0;
  //  else if(Int)a > 100) return gammpapprox(a,x,0);
  else if (x < a+1.0) return 1.0-gser(a,x);
  else return gcf(a,x);
}
Doub pks(Doub z) { // Kolmogorov-Smirnov distribution
  if (z < 0.) throw("bad z in KSdist");
  if (z < 0.042) return 0.;
  if (z < 1.18) {
    Doub y = exp(-1.23370055013616983/SQR(z));
    return 2.25675833419102515*sqrt(-log(y))
      *(y + pow(y,9) + pow(y,25) + pow(y,49));
  } else {
    Doub x = exp(-2.*SQR(z));
    return 1. - 2.*(x - pow(x,4) + pow(x,9));
  }
}
Doub qks(Doub z) {
  if (z < 0.) throw("bad z in KSdist");
  if (z == 0.) return 1.;
  if (z < 1.18) return 1.-pks(z);
  Doub x = exp(-2.*SQR(z));
  return 2.*(x - pow(x,4) + pow(x,9));
}
	
// Doub Erf::inverfc(Doub p) {
// 		Doub x,err,t,pp;
// 		if (p >= 2.0) return -100.;
// 		if (p <= 0.0) return 100.;
// 		pp = (p < 1.0)? p : 2. - p;
// 		t = sqrt(-2.*log(pp/2.));
// 		x = -0.70711*((2.30753+t*0.27061)/(1.+t*(0.99229+t*0.04481)) - t);
// 		for (Int j=0;j<2;j++) {
// 			err = erfc(x) - pp;
// 			x += err/(1.12837916709551257*exp(-SQR(x))-x*err);
// 		}
// 		return (p < 1.0? x : -x);
// 	}

//	inline Doub inverf(Doub p) {return inverfc(1.-p);}


// struct Normaldist : Erf {
// 	Doub mu, sig;
// 	Normaldist(Doub mmu = 0., Doub ssig = 1.) : mu(mmu), sig(ssig) {
// 		if (sig <= 0.) throw("bad sig in Normaldist");
// 	}
// 	Doub p(Doub x) {
// 		return (0.398942280401432678/sig)*exp(-0.5*SQR((x-mu)/sig));
// 	}
// 	Doub cdf(Doub x) {
// 		return 0.5*erfc(-0.707106781186547524*(x-mu)/sig);
// 	}
// 	Doub invcdf(Doub p) {
// 		if (p <= 0. || p >= 1.) throw("bad p in Normaldist");
// 		return -1.41421356237309505*sig*inverfc(2.*p)+mu;
// 	}
// };

// end NR code
void Welford::update(double x, double weight){
  if(weight <= 0) return; 
  double delta = (W==0? weight*x : weight*(x-S1/W));
  W = tau*W + weight;
  S1 = tau*S1 + weight*x;
  S2 = tau*S2 + delta*(x-S1/W);
}


// LCG parameters from Wikipedia
const uint64_t LCG64::a = 2862933555777941757;
const uint64_t LCG64::b = 3037000493;
const double LCG64::uint64_max = (double)UINT64_MAX;

void deblank(char* p){ // eliminate ' ' and '\t' from string
  char* q = p;

  while((*p = *q++)){
    if(!isblank(*p))p++;
  }
} 

PopCount::PopCount(void){
  for(int i = 0;i < 256;i++){ // initialize byte counts
    byte_count[i] = 0;
    for(int j = 1;j < 256;j<<= 1) if(i & j) byte_count[i]++;
  }
}
// uint PopCount::operator()(uint64_t word){
//   uint density = 0;
//   for(int i = 0;i < 8;i++){
//     density += byte_count[word&0xff];
//     word >>= 8;
//   }
//   return density;
// }
uint PopCount::operator()(uint64_t word){
  uint density = byte_count[word&0xff];
  word >>= 8;
  density += byte_count[word&0xff];
  word >>= 8;
  density += byte_count[word&0xff];
  word >>= 8;
  density += byte_count[word&0xff];
  word >>= 8;
  density += byte_count[word&0xff];
  word >>= 8;
  density += byte_count[word&0xff];
  word >>= 8;
  density += byte_count[word&0xff];
  word >>= 8;
  density += byte_count[word&0xff];
  return density;
}
      

// int StringSplitter::operator()(char* record, char fs){
//   nf = 1;
//   field.clear();
//   while(isspace(*record)record++;  //skip initial whitespace
//   field.push_back(record); // set the first pointer
//   while(*record){
//     if( (fs == ' ' && isspace(*record)) || *record == fs){ // terminate string and store next start position
//       *record++ = '\0'; // replace fs with \0
//       while(isspace(*record))record++; // skip leading whitespace
//       if(*record){
//         field.push_back(record);
//         nf++;
//       }
//     }
//     else record++;
//   }
//   return nf;
// }

int StringSplitter::operator()(char* record, char fs){
  nf = 0;
  do{
    while(isspace(*record))record++;  //skip leading whitespace
    if(*record){
      if(nf >= MAXFIELDS) throw "StringSplitter: String has too many fields\n";
      field[nf++] = record;  // set the next pointer
    }
    while(*record && *record != fs)record++; // skip forward to next field sep or eol
    if(*record) *record++ = '\0'; // replace fs with '\0'
  }while(*record);
  return nf;
}

char* StringSplitter::operator[](int i){
  if(i < 0 || i >= nf) return NULL;
  return field[i];
}


void split_string(char* line, std::vector<char*>& token, char fs){ // input: char* line with
  //fields delimited by fs, and vector<char*> token, an empty array  of c-strings.
  //output: '\0' replaces fs, making line a sequence of c-strings whose addresses are stored 
  //in token.
  token.clear();
  token.push_back(line);
  while(*line){
    if(*line == fs){
      *line++ = '\0';
      token.push_back(line);
    }
    else line++;
  }
}



// quick and dirty printf-type format string adapter for c++ streams
// usage, e.g:  cout << format("Now we can do formatted output such as six = %d, pi = %f, etc",6, 3.14159)<<endl;
// set MAXCHARS appropriately 

std::string format(const char* fstr,...){
  char buf[MAXCHARS+1]; // will truncate after MAXCHARS characters -- no buffer overflows
  va_list ptr;
  //int nchars;

  va_start(ptr,fstr);
  int nchars =  vsnprintf(buf,MAXCHARS,fstr,ptr);
  if(nchars >= MAXCHARS) throw "format: too many characters!\n";
  //cout << "format: vsnprintf returns "<< nchars<<endl;
  va_end(ptr);
  std::string output(buf);
  return output;
}

//***************************

#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
std::string currentDateTime(void) {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}

std::string str_time(double t){
  int it = (int)t;
  time_t tt = it;
  double ft = t-it;
  const char* fmt = "%.2d:%.2d:%7.4f";

  struct tm* pl = localtime(&tt);
  //std::cout << "localtime sets isdst to "<<pl->tm_isdst<<std::endl;
  return format(fmt,pl->tm_hour,pl->tm_min,pl->tm_sec + ft);
}

std::string str_date(double t){
  time_t tt = (int)t;
  struct tm* pl = localtime(&tt);
  std::ostringstream s;
  
  s <<  pl->tm_mon+1 <<"/"<<pl->tm_mday<<"/"<<pl->tm_year+1900;
  return s.str();
}

double get_time(int date, std::string time){
  std::stringstream ss;
  struct tm pl;
  char buf[18];


  ss.str("");
  ss <<date<<time<<":00";
  strncpy(buf,ss.str().c_str(),18);
  //std::cout << "calling strptime with "<<buf<<std::endl;

  strptime(buf,"%Y%m%d%H:%M:%S",&pl);
  //std::cout <<"tm: "<<pl.tm_sec<<" "<<pl.tm_min<<" "<<pl.tm_hour<<" "<<pl.tm_mday<<" "<<pl.tm_mon<<" "<<pl.tm_year<<" "<<pl.tm_wday<<" "<<pl.tm_yday<<" "<<pl.tm_isdst<<std::endl;
  //std::cout << "get_time has isdst = "<<pl.tm_isdst<<std::endl;
  pl.tm_isdst = -1; // fixing a bug (design defect?) in strptime
  strftime(buf,sizeof(buf), "%s", &pl);
  //std::cout << "mktime returns "<<mktime(&pl)<<" strftime has "<<buf<<std::endl;
  return atof(buf);
}

uint32_t String::hash1(uintptr_t salt) const{ // simple hash function
  int i,j;
  uint32_t sum = 0;
  const char* s = this->c_str();
  int n = strlen(s);
  int m = (n < 4? 4:n);
  for(i = j = 0; i < m;i++){
    if(j == 0) sum ^= salt<<i;
    unsigned char c = (i >= n ? ' ': s[i]); // pad short strings with blanks
    sum += c <<(8*j);
    j = (j+1)%4;
    //      cout << format("s[%d] = %c, sum = %d\n",i,s[i],sum)<<endl;
  }
  //  std::cout << this<<" hashing "<<s<<" to "<<sum<<std::endl;

  return sum;
}

String String::null("");

uint32_t str_hash(const char* s, uintptr_t salt){
  uint32_t b;
  uint8_t* b0 = (uint8_t*)&b;
  int j0,j1,j2,j3;
  /*  b0[0] = salt&0xff;
  b0[1] = (salt>>8)&0xff;
  b0[2] = (salt>>16)&0xff;
  b0[3] = (salt>>24)&0xff;
  */
  b = salt;

  for(int i = 0;s[i];i++){ // mix in s
    j0 = i&3;
    j1 = (i+1)&3;
    j2 = (i+2)&3;
    j3 = (i+3)&3;
    b0[j0] += (uint8_t)s[i] + b0[j3]*b0[j2] + b0[j1];
  }
  for(int i = 0;i < 8;i++){ // additional mixing
    j0 = i&3;
    j1 = (i+1)&3;
    j2 = (i+2)&3;
    j3 = (i+3)&3;
    b0[j3] += (b0[j0]*b0[j1]) + b0[j2];
  }
  //std::cout << " hashing "<<s<<" to "<<b<<std::endl;
  
  return b;
}
// double interp(Array<IndexPair<double,double>> table, double x, int& upper, int& lower){
//   /* IndexPair is defined in util.h
//    * At entry, we must have table[i].item1 < table[i+1].item1 for all i, and lower < upper.
//    * The table will be searched only between table[lower].item1 and table[upper].item1
//    *
//    * usage: First set upper and lower (e.g. lower = 0; upper = table.len()-1).
//    * If x = table[i].item1 for some i, table[i].item2 is returned with upper = i.  
//    * If x >= table[upper].item1, table[upper].item2 is returned.
//    * If x <= table[lower].item1, table[lower].item2 is returned.
//    * Otherwise, at return we set upper = lower+1, l := table[lower].item1 < x < u := table[upper].item1,
//    * and return (1-f)*table[lower].item2 + f*table[upper].item2, where f = (x - l)/(u - l).
//    */
//   int new_bound;
//   assert(lower < upper);
//   if(table[upper].item1 <= x) return table[upper].item2;
//   if(x <= table[lower].item1) return table[lower].item2;

//   // now table[lower].item1 < x <= table[upper].item1 and those inequalities are preserved
//   // in the loop
//   while(upper - lower > 1){
//     if(table[new_bound = (lower+upper)/2].item1 < x) lower = new_bound;
//     else upper = new_bound;
//   }
//   double f = (x - table[lower].item1)/(table[upper].item1 - table[lower].item1);
//   return (1-f)*table[lower].item2 + f*table[upper].item2;
// }

// int dedup (const Array<double>& value, Array<IndexPair<double,double>>& pairs){
//   /* This function finds the unique entries in the sorted value vector and
//    * copies them to pairs.item1, placing their fractional rank (quantile value)
//    * into pairs.item2.
//    */
//   int j = 0;
//   int n = value.len();
//   int i1 = 0;
//   for(int i = 0;i < n;i++){
//     i1 = i;
//     while(i < n-1 && value[i+1] == value[i])i++;
//     pairs[j].item1 = value[i]; // x_i
//     pairs[j++].item2 = (i+i1)/(2.0*n); // if there's a run of repeated values, use the mean index value
//   }
//   return j;
// }


template<typename T1, typename T2>
int IndexPair<T1,T2>::sort_item; // defaults to first item
