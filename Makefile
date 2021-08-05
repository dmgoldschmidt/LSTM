INC = /home/david/code/include
SRC = /home/david/code/src
CFLAGS = -g -Wall -I$(INC) -std=c++11

util.o: $(INC)/util.h $(SRC)/util.cc
	g++ $(CFLAGS) -c -o util.o $(SRC)/util.cc 
GetOpt.o: $(INC)/GetOpt.h $(SRC)/GetOpt.cc
	g++ $(CFLAGS) -c -o GetOpt.o $(SRC)/GetOpt.cc
Awk.o:	Awk.h Awk.cc util.o 
	g++ $(CFLAGS) -c -o Awk.o Awk.cc
test_gz: test_gzstream.cc gzstream/gzstream.o Awk.o util.o GetOpt.o
	g++ $(CFLAGS) -o test_gz test_gzstream.cc gzstream/gzstream.o Awk.o util.o GetOpt.o -lz
read_lanl: read_lanl.cc gzstream/gzstream.o Awk.o util.o GetOpt.o matrix.o GammaFunc.o Array.h KeyIndex.h Histogram.h GammaFunc.h
	g++ $(CFLAGS) -o read_lanl read_lanl.cc gzstream/gzstream.o Awk.o util.o GetOpt.o matrix.o GammaFunc.o -lz
matrix.o: matrix.cc Matrix.h Array.h
	g++ $(CFLAGS) -c matrix.cc
digamma.o: digamma.cc
	g++ $(CFLAGS) -c digamma.cc
GammaFunc.o: GammaFunc.cc GammaFunc.h
	g++ $(CFLAGS) -c GammaFunc.cc
fit_gamma: fit_gamma.cc digamma.o util.o GetOpt.o Awk.o matrix.o GammaFunc.o Matrix.h Array.h  GetOpt.h util.h GammaFunc.h
	g++ $(CFLAGS) -o fit_gamma fit_gamma.cc digamma.o util.o GetOpt.o Awk.o matrix.o gzstream/gzstream.o GammaFunc.o -lz
hashstats: hashstats.cc Awk.o util.o GetOpt.o KeyIndex.h Array.h
	g++ $(CFLAGS) -o hashstats hashstats.cc Awk.o util.o GetOpt.o gzstream/gzstream.o -lz 
test_PopCount: test_PopCount.cc util.o
	g++ $(CFLAGS) -o test_PopCount test_PopCount.cc util.o
bloom_capacity: bloom_capacity.cc
	g++ $(CFLAGS) -o bloom_capacity bloom_capacity.cc
EM.o: EM.cc EM.h
	g++ $(CFLAGS) -c EM.cc
testEM: testEM.cc EM.o GetOpt.o util.o Matrix.h
	g++ $(CFLAGS) -o testEM testEM.cc EM.o GetOpt.o util.o 
Trigraphic.o: Trigraphic.cc Trigraphic.h
	g++ $(CFLAGS) -c Trigraphic.cc
PersistentState.o: PersistentState.cc PersistentState.h KeyIndex.h BlockArray.h util.h
	g++ $(CFLAGS) -c PersistentState.cc
model_names: model_domain_names.cc Trigraphic.o util.o Awk.o GetOpt.o PersistentState.o 
	g++ $(CFLAGS) -o model_names model_domain_names.cc Trigraphic.o PersistentState.o util.o Awk.o GetOpt.o gzstream/gzstream.o -lz -lboost_regex
ps_main: PersistentState.h PersistentState.cc util.o GetOpt.o
	g++ $(CFLAGS) -DPS_MAIN -o ps_main PersistentState.cc util.o GetOpt.o -lboost_regex 
DnsMonitor.o: DnsMonitor.cc DnsMonitor.h Heap.h Array.h KeyIndex.h Awk.h util.h
	g++ $(CFLAGS) -c DnsMonitor.cc
testDnsMonitor: testDnsMonitor.cc DnsMonitor.o Trigraphic.o util.o Awk.o GetOpt.o EM.o PersistentState.o Cstrings.o
	g++ $(CFLAGS) -o testDnsMonitor testDnsMonitor.cc DnsMonitor.o Trigraphic.o EM.o util.o Awk.o GetOpt.o PersistentState.o Cstrings.o gzstream/gzstream.o -lz -lboost_regex
test2models: test2models.cc Trigraphic.o util.o Awk.o GetOpt.o PersistentState.o 
	g++ $(CFLAGS) -o test2models test2models.cc Trigraphic.o PersistentState.o util.o Awk.o GetOpt.o gzstream/gzstream.o -lz -lboost_regex
check_model: check_model.cc Trigraphic.o Awk.o util.o  GetOpt.o PersistentState.o 
	g++ $(CFLAGS) -o check_model check_model.cc Trigraphic.o Awk.o PersistentState.o util.o GetOpt.o gzstream/gzstream.o -lz  -lboost_regex 
Cstrings.o: Cstrings.h Cstrings.cc util.o Array.h KeyIndex.h
	g++ $(CFLAGS) -c Cstrings.cc
testCstrings: testCstrings.cc Cstrings.o util.o
	g++ $(CFLAGS) -o testCstrings testCstrings.cc util.o Cstrings.o
update_suffix_tree: update_suffix_tree.cc SuffixTree.h util.o GetOpt.o Awk.o PersistentState.o 
	g++ $(CFLAGS) -o update_suffix_tree update_suffix_tree.cc util.o GetOpt.o gzstream/gzstream.o Awk.o PersistentState.o -lz -lboost_regex
test_suffix_tree:  update_suffix_tree.cc SuffixTree.h util.o GetOpt.o Awk.o PersistentState.o 
	g++ $(CFLAGS) -DTEST -o test_suffix_tree update_suffix_tree.cc util.o GetOpt.o gzstream/gzstream.o Awk.o PersistentState.o -lz -lboost_regex
score_names.o: score_names.cc score_names.h Trigraphic.h SuffixTree.h PersistentState.h util.h 
	g++ $(CFLAGS) -c score_names.cc
test_score_names:  score_names.cc score_names.h GetOpt.o util.o Awk.o SuffixTree.h PersistentState.o Trigraphic.o  
	g++ $(CFLAGS) -DMAIN -o test_score_names score_names.cc GetOpt.o Awk.o util.o PersistentState.o Trigraphic.o gzstream/gzstream.o -lz -lboost_regex
netflow_stats: netflow_stats.cc Awk.o Array.h Heap.h GetOpt.o util.o
	g++ $(CFLAGS) -o netflow_stats netflow_stats.cc GetOpt.o Awk.o util.o gzstream/gzstream.o -lz 
FB.o: FB.h FB.cc GetOpt.o util.o Matrix.h Array.h
	g++ $(CFLAGS) -c FB.cc
test_FB: FB.o GetOpt.o util.o Matrix.h Array.h
	g++ $(CFLAGS) -DMAIN -o test_FB test_FB.cc FB.o util.o GetOpt.o  
BF.o: BF.h BF.cc GetOpt.o util.o Matrix.h Array.h
	g++ $(CFLAGS) -c BF.cc
test_BF: BF.h BF.cc GetOpt.o util.o Matrix.h Array.h
	g++ $(CFLAGS) -DMAIN -o test_BF BF.cc util.o GetOpt.o  
fb_dag: fb_dag.cc GetOpt.o util.o Matrix.h
	g++ $(CFLAGS) -o fb_dag fb_dag.cc GetOpt.o util.o
BLreg.o: BLreg.h BLreg.cc Array.h Matrix.h QRreg.h PersistentState.o
	g++ $(CFLAGS) -c BLreg.cc
QRreg.o: QRreg.h QRreg.cc Array.h Matrix.h util.h util.cc
	g++ $(CFLAGS) -c QRreg.cc
test_BLreg: BLreg.cc BLreg.h Array.h Matrix.h  QRreg.o PersistentState.o GetOpt.o util.o
	g++ $(CFLAGS) -DBL_MAIN -o test_BLreg BLreg.cc QRreg.o PersistentState.o GetOpt.o util.o -lboost_regex
stats: stats.cc QRreg.o GetOpt.o util.o Array.h Awk.o matrix.o
	g++ $(CFLAGS) -o stats stats.cc QRreg.o GetOpt.o util.o Awk.o matrix.o gzstream/gzstream.o -lz
HMM.o: HMM.h HMM.cc Matrix.h Array.h matrix.o
	g++ $(CFLAGS) -c HMM.cc  
testHMM: testHMM.cc HMM.o GetOpt.o util.o Array.h Matrix.h Awk.o QRreg.o
	g++ $(CFLAGS) -o testHMM testHMM.cc HMM.o GetOpt.o util.o Awk.o QRreg.o matrix.o gzstream/gzstream.o -lz
HMMsim: HMMsim.cc HMM.o GetOpt.o util.o Array.h Matrix.h Awk.o 
	g++ $(CFLAGS) -o HMMsim HMMsim.cc HMM.o GetOpt.o util.o Awk.o QRreg.o matrix.o gzstream/gzstream.o -lz
ProbHistogram.o: ProbHistogram.h ProbHistogram.cc
	g++ $(CFLAGS) -c ProbHistogram.cc	
MarketHMM: MarketHMM.cc HMM.o GetOpt.o util.o Array.h Matrix.h Gaussian.h Awk.o ProbHistogram.o QRreg.o 
	g++ $(CFLAGS) -o MarketHMM MarketHMM.cc HMM.o GetOpt.o util.o Awk.o matrix.o ProbHistogram.o QRreg.o \
gzstream/gzstream.o -lz
qr_reg: qr_reg.cc QRreg.o GetOpt.o util.o
	g++ $(CFLAGS) -o qr_reg qr_reg.cc QRreg.o util.o GetOpt.o 
LSTM.o: LSTM.h LSTM.cc $(INC)/Matrix.h $(INC)/Array.h
	g++ $(CFLAGS) -c LSTM.cc
test_LSTM: LSTM.o test_LSTM.cc util.o $(INC)/Matrix.h $(INC)/Array.h GetOpt.o
	g++ $(CFLAGS) -o test_LSTM test_LSTM.cc LSTM.o util.o GetOpt.o



