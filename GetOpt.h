#ifndef GETOPT_H
#define GETOPT_H
#include<iostream>
#include<vector>
#include<string>
#include<cstring>
#include "using.h"

/* Command line parser
 * The constructor creates two n-vectors of strings.  It assumes there are n (option, argument)
 * pairs, with the options prefaced by '-' for single-letter options, or "--" for multi-letter
 * options.  Any field not prefaced by at least one '-' is an argument.  If an option or an argument 
 * is missing, the corresponding string is "".  So arguments without any corresponding option can appear 
 * anywhere, as can options without an argument.  The isolated string "-" is parsed as an argument.
   
 * There is no need to tell the parser what the expected options are, to make them be single
 * characters, or to place them all before any arguments without options.  This syntax is mostly compatible 
 * with the usual syntax, with the following exception: an option-less argument occurs if and only if it 
 * follows another argument, and an argument-less option occurs if and only if either
 * a) it precedes another option, or b) it is the last entry on the command line.  So if the user wants to specifiy
 * an argument-less option like "-v" followed immediately by an option-less argument like "1.2", say, the sequence
 *  "-v 1.2" on the command line will not be parsed as two singletons. This can be easily fixed by supplying a 
 * dummy argument to -v.
 * The long option style "--file=myfile" is supported, but in addition the equals sign is superfuous.  Equivalently,
 * you can write "--file myfile".  Long argument-less options are fine, e.g. "--verbose".

 * The convenience method GetOpt::get(char const* opt) will search for the option "opt" and return the number of matches
 * it finds.  If it finds exactly one and there is a second argument, it sets the second argument to the
 * matching argument on the command line, performing the conversion dictated by the type of the second argument.
 * This can produce undefined results if either the argument is missing or it has the wrong type.

 * Get will match any prefix of opt that appears as an option on the command line, hence a long first argument can be 
 * abbreviated by any of its (unique) prefixes.  But to be safe, the programmer should check the return value of the
 * get call because it might be > 1, for example if the command line says "-f file --first=mystring", then querying for
 * --first will return two matches.

 * The method const char* GetOpt::get(int n) will return a pointer to the nth option-less argument (reading left to
 * right and counting 1-up) or nullptr if there are fewer than n option-less arguments.

 * If the option "-help" appears anywhere on the command line, the help_msg, which defaults to "No help available", 
 * is printed and exit(0) is called.

 * If a non-null comment character is given to the constructor, the command line prefaced by that character will be 
 * written to stdout.

*/

class GetOpt{
  int j;
public:
  vector<string> option;
  vector<string> argument;

  GetOpt(int argc, char** argv, char const* help_msg = "No help available\n", char comment = '#');
  const char* get(int n);// return nth optionless argument (0-up), or nullptr if it doesn't exist
  int get(char const* opt); // return nmatches = no. of matches for opt
  int get(char const* opt, int& arg);   // return nmatches  and set integer arg iff nmatches == 1
  int get(char const* opt, double& arg);// ditto for double 
  int get(char const* opt, float& arg);// ditto for float 
  int get(char const* opt, string& arg);// ditto for string
  int get(char const* opt, char& arg); // ditto for char
};
#endif
