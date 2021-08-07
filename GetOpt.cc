#include "GetOpt.h"
#include "util.h"

GetOpt::GetOpt(int argc, char** argv, char const* help_msg, char comment){ // parse command line
  bool opt_read = false; // parser state

  for(int i = 1;i < argc;i++){
    if(*argv[i] == '-' && !isdigit(*(argv[i]+1)) && *(argv[i]+1) != '.' && *(argv[i]+1) != '\0'){
      // it's an option
      if(opt_read)argument.push_back(""); // two options in a row. strore empty argument
      argv[i]++; // skip the option flag
      if(strcmp("help",argv[i]) == 0){ // "-help" prints the help_msg and exits
        cout << help_msg;
        exit(0);
      }
      char* p = nullptr;
      if(*argv[i] == '-'){ // long-style multi-letter option or "option=argument"
        argv[i]++; // skip the double minus
        p = strchr(argv[i],'='); // find the equals sign or nullptr if no equals sign 
        if(p != nullptr) *p++ = '\0';// replace it with EOS and move to the argument
        option.push_back(argv[i]); // store the long-style option
      }
      else{// it's a single letter option
        if(strlen(argv[i]) > 1){ // old-style single letter option followed immediately by argment
          char old_opt[2];
          old_opt[0] = *argv[i];
          old_opt[1] = '\0';
          option.push_back(&old_opt[0]);
          p = argv[i]+1; 
        }
        else option.push_back(argv[i]); // store the single letter option
      }
      if(p != nullptr){ // the argument was included in the same command-line field
        argument.push_back(p);
        opt_read = false;
      }
      else opt_read = true; // it was an option only (including the case --option with no "=argument")
    }
  
    else{ // it's an argument (possibly a negative number or an isolated -
      if(!opt_read)option.push_back(""); // two arguments in a row. store empty option
      argument.push_back(argv[i]);
      opt_read = false;
    }
  }
  if(opt_read)argument.push_back(""); // we ended with an option
  if(1){
    for(int i = 0;i < option.size();i++) cout << option[i]<<": "<<argument[i]<<endl;
  }
  if(comment) {
    cout << comment<< currentDateTime()<<": ";
    for(int i = 0;i < argc;i++) cout <<argv[i]<<" ";
    cout <<endl;
    std::flush(cout);
  }

}

// Answer queries
int GetOpt::get(char const* opt){ // look for option
  uint i,l;
  int nmatches = 0;

  for(i = 0;i < option.size();i++){
    l = strlen(option[i].c_str());
    if(l != 0 && strncmp(opt,option[i].c_str(),l) == 0){
      nmatches++;
      j = i; // store position in option list
    }
  }
  return nmatches;
}

const char* GetOpt::get(int n){ // return nth optionless argument (0-up), or nullptr if it doesn't exist
  if(n <= 0) return nullptr;
  for(int i = 0;i < option.size();i++){
    if(option[i] == "")n--;
    if(n == 0) return argument[i].c_str();
  }
  return nullptr;
}
    


int GetOpt::get(char const* opt, int& arg){ // return integer argument
  int nmatches = get(opt);

  if(nmatches == 1) arg = atoi(argument[j].c_str());
  return nmatches;
}

int GetOpt::get(char const* opt, double& arg){ // return floating pt. argument
  int nmatches = get(opt);

  if(nmatches == 1) arg = atof(argument[j].c_str());
  return nmatches;
}

int GetOpt::get(char const* opt, float& arg){ // return floating pt. argument
  int nmatches = get(opt);

  if(nmatches == 1) arg = atof(argument[j].c_str());
  return nmatches;
}

int GetOpt::get(char const* opt, string& arg){// return string argument
  int nmatches = get(opt);

  if(nmatches == 1) arg = argument[j];
  return nmatches;
}

int GetOpt::get(char const* opt, char& arg){ // return char argument
  int nmatches = get(opt);
  
  if(nmatches == 1) arg = argument[j][0];
  return nmatches;
}


  
  
#ifdef GetOpt_MAIN
int main(int argc, char**argv){
  GetOpt cmd_line(argc,argv, "testing help msg.\n");
  int a = cmd_line.argument.size();
  int o = cmd_line.option.size();
  int i_arg; // test integer argument (set for option = "int" below)
  double d_arg; // test double argument (option = "double")
  string s_arg; // test string argument (option = "string")
  // NOTE:  you can use any option names you want

  cout << o << " options:\n";
  for(int i = 0;i < o;i++)cout << cmd_line.option[i]<<endl;
  cout<<endl << a << " arguments:\n";
  for(int i = 0;i < a;i++)cout << cmd_line.argument[i]<<endl;
  if(cmd_line.get("int",i_arg) == 1) cout << "i_arg = "<< i_arg<<endl;
  if(cmd_line.get("double",d_arg) == 1) cout <<"d_arg = "<<d_arg<<endl;
  if(cmd_line.get("string",s_arg) == 1) cout <<"s_arg = "<<s_arg<<endl;
}
#endif
