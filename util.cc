#include "util.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cctype>

using namespace std;

void Welford::update(double x, double weight){
  if(weight <= 0) return; 
  double delta = (W==0? weight*x : weight*(x-S1/W));
  W = tau*W + weight;
  S1 = tau*S1 + weight*x;
  S2 = tau*S2 + delta*(x-S1/W);
}

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
	
Doub Erf::inverfc(Doub p) {
  Doub x,err,t,pp;
  if (p >= 2.0) return -100.;
  if (p <= 0.0) return 100.;
  pp = (p < 1.0)? p : 2. - p;
  t = sqrt(-2.*log(pp/2.));
  x = -0.70711*((2.30753+t*0.27061)/(1.+t*(0.99229+t*0.04481)) - t);
  for (Int j=0;j<2;j++) {
    err = erfc(x) - pp;
    x += err/(1.12837916709551257*exp(-x*x)-x*err);
  }
  return (p < 1.0? x : -x);
}
const Doub Erf::cof[28] = {-1.3026537197817094, 6.4196979235649026e-1,
                           1.9476473204185836e-2,-9.561514786808631e-3,-9.46595344482036e-4,
                           3.66839497852761e-4,4.2523324806907e-5,-2.0278578112534e-5,
                           -1.624290004647e-6,1.303655835580e-6,1.5626441722e-8,-8.5238095915e-8,
                           6.529054439e-9,5.059343495e-9,-9.91364156e-10,-2.27365122e-10,
                           9.6467911e-11, 2.394038e-12,-6.886027e-12,8.94487e-13, 3.13092e-13,
                           -1.12708e-13,3.81e-16,7.106e-15,-1.523e-15,-9.4e-17,1.21e-16,-2.8e-17};
Doub erfcc(const Doub x)
{
  Doub t,z=fabs(x),ans;
  t=2./(2.+z);
  ans=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
      t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
      t*(-0.82215223+t*0.17087277)))))))));
  return (x >= 0.0 ? ans : 2.0-ans);
}

Doub Normaldev::dev(void) {
  Doub u,v,x,y,q;
  do {
    u = uniform();
    v = 1.7156*(uniform()-0.5);
    x = u - 0.449871;
    y = abs(v) + 0.386595;
    q = x*x + y*(0.19600*y-0.25472*x);
  } while (q > 0.27597
           && (q > 0.27846 || v*v > -4.*log(u)*u*u));
  return mu + sig*v/u;
}


// LCG parameters from Wikipedia
const uint64_t LCG64::a = 2862933555777941757;
const uint64_t LCG64::b = 3037000493;
const double LCG64::uint64_max = (double)UINT64_MAX;

void deblank(char* p){ // eliminate ' ' and '\t' from string
  char* q = p;

  while(*p = *q++){
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
