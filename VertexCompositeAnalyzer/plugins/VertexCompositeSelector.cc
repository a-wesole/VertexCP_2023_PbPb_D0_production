// system include files
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>

#include <TF2.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TTree.h>
#include <TFile.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TString.h>
#include <TObjString.h>
#include <TCanvas.h>
#include <TVector3.h>
#include <TMatrixD.h>
#include <TRandom.h>
#include <TMath.h>

// user include files

#include "DataFormats/Common/interface/Ref.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"


#include "DataFormats/TrackReco/interface/DeDxData.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include "BDT_header.h"
//#include "BDT_header_pt_trainings.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
//
// class decleration
//

#define PI 3.1416

const int y_bins = 2;
const int cent_bins = 4;
const int pT_bins = 10;

using namespace std;
using namespace cms::Ort;

using namespace std;

class BDTHandler
{
private:
  double bdtCuts[y_bins][cent_bins][pT_bins];

    tuple<int, int, int> get_bins(float y, int centrality, float pT) const
  {
      int y_bin = -1, cent_bin = -1, pT_bin = -1;

      if (centrality >= 0 && centrality < 2 * 10) cent_bin = 0;
      else if (centrality >= 2 * 10 && centrality < 2 * 30) cent_bin = 1;
      else if (centrality >= 2 * 30 && centrality < 2 * 50) cent_bin = 2;
      else if (centrality >= 2 * 50 && centrality < 2 * 90) cent_bin = 3;

      if (pT >= 1 && pT < 2) pT_bin = 0;
      else if (pT >= 2 && pT < 3) pT_bin = 1;
      else if (pT >= 3 && pT < 4) pT_bin = 2;
      else if (pT >= 4 && pT < 5) pT_bin = 3;
      else if (pT >= 5 && pT < 6) pT_bin = 4;
      else if (pT >= 6 && pT < 8) pT_bin = 5;
      else if (pT >= 8 && pT < 10) pT_bin = 6;
      else if (pT >= 10 && pT < 15) pT_bin = 7;
      else if (pT >= 15 && pT < 20) pT_bin = 8;
      else if (pT >= 20) pT_bin =9;

      if (abs(y) < 1) y_bin = 0;
      else if (abs(y) >= 1 && abs(y) < 3) y_bin = 1;

      return make_tuple(y_bin, cent_bin, pT_bin);
  }

public:
  BDTHandler(const string &filename)
  {
    loadCuts(filename);
  }

  void loadCuts(const string &filename)
  {
    ifstream file(filename);
    if (!file.is_open()) cerr << "error no csv file" << endl;
    string line;

    while (getline(file, line))
    {
      if (line[0] == '#') continue;
      stringstream ss(line);
      int y_bin =-9, cent_bin=-9, pT_bin=-9;
      double cut_value;

      ss >> y_bin >> cent_bin >> pT_bin >> cut_value;

      bdtCuts[y_bin][cent_bin][pT_bin] = cut_value; // writes bdt cut values to the bdCuts array
    }
    file.close();
  }

  inline double getBDTCut(float y, int centrality, float pT) const
  {
    int y_bin, cent_bin, pT_bin;
    tie(y_bin, cent_bin, pT_bin) = get_bins(y, centrality, pT);
    if (y_bin < 0 || cent_bin < 0 || pT_bin < 0) {
      return 999;
    }
    return bdtCuts[y_bin][cent_bin][pT_bin];
  }
};

class VertexCompositeSelector : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>>
{

	public:
		explicit VertexCompositeSelector(const edm::ParameterSet &, const ONNXRuntime *cache);
		static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &);
		static void globalEndJob(const ONNXRuntime *);
		~VertexCompositeSelector();


  using MVACollection = std::vector<float>;

private:
  virtual void beginJob();
  virtual void produce(edm::Event &, const edm::EventSetup &);
  void fillRECO(edm::Event &iEvent, const edm::EventSetup &iSetup);
  virtual void endJob();


		std::vector<std::string> theInputVars;
		vector<double> inputValues;
		ReadBDT *mva;
		BDTHandler bdt;
		std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
		std::vector<std::vector<int64_t>> input_shapes_;
		// FloatArrays data_; 
		std::string onnxModelPath_;
        const ONNXRuntime* onnxRuntime_ ;

  // ----------member data ---------------------------

  // options

  int PID_;
  int PID_dau1_;
  int PID_dau2_;

  // cut variables
  double multMax_;
  double multMin_;
  double deltaR_; // deltaR for Gen matching
  bool trkHighPurity_;
  double trkPMin_;
  double trkPtMin_;
  double trkEtaMax_;
  double trkPSumMin_;
  double trkPtSumMin_;
  double trkPtAsymMin_;
  double trkEtaDiffMax_;
  double trkPtErrMax_;
  int trkNHitMin_;
  double candpTMin_;
  double candpTMax_;
  double candYMin_;
  double candYMax_;
  double cand3DDecayLengthSigMin_;
  double cand2DDecayLengthSigMin_;
  double cand3DPointingAngleMax_;
  double cand2DPointingAngleMax_;
  double cand3DDCAMin_;
  double cand3DDCAMax_;
  double cand2DDCAMin_;
  double cand2DDCAMax_;
  double candVtxProbMin_;

  double mvaCut_;

  // tree branches
  // event info
  int centrality;
  int Ntrkoffline;
  float bestvx;
  float bestvy;
  float bestvz;

  // Composite candidate info
  float mva_value;
  float mva_old;
  float pt;
  float eta;
  float flavor;
  float y;
  float mass;
  float VtxProb;
  float dlos;
  float dl;
  float dlerror;
  float agl;
  float vtxChi2;
  float ndf;
  float agl_abs;
  float agl2D;
  float agl2D_abs;
  float dlos2D;
  bool isSwap;
  bool matchGEN;
  int idmom_reco;

  // dau info
  float dzos1;
  float dzos2;
  float dxyos1;
  float dxyos2;
  float nhit1;
  float nhit2;
  bool trkquality1;
  bool trkquality2;
  float pt1;
  float pt2;
  float ptErr1;
  float ptErr2;
  float p1;
  float p2;
  float eta1;
  float eta2;
  float phi1;
  float phi2;
  float H2dedx1;
  float H2dedx2;
  float T4dedx1;
  float T4dedx2;
  float trkChi1;
  float trkChi2;
  bool isPionD1;
  bool isPionD2;
  bool isKaonD1;
  bool isKaonD2;

  int selectFlavor_;
  bool usePID_;
  bool useAnyMVA_;

  bool assignBDT = true;

  std::vector<float> mvaVals_;

  TF2 *func_mva;
  std::vector<double> mvaCuts_;

  TH2D *hist_bdtcut;

  float mvaMin_;
  float mvaMax_;

  int centMin_;
  int centMax_;
  bool isCentrality_;

  edm::Handle<int> cbin_;

  // tokens
  edm::EDGetTokenT<reco::VertexCollection> tok_offlinePV_;
  edm::EDGetTokenT<reco::TrackCollection> tok_generalTrk_;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> patCompositeCandidateCollection_Token_;
  edm::EDGetTokenT<MVACollection> MVAValues_Token_;
  edm::EDGetTokenT<edm::ValueMap<reco::DeDxData>> Dedx_Token1_;
  edm::EDGetTokenT<edm::ValueMap<reco::DeDxData>> Dedx_Token2_;
  edm::EDGetTokenT<reco::GenParticleCollection> tok_genParticle_;
  edm::EDGetTokenT<int> tok_centBinLabel_;
  edm::EDGetTokenT<reco::Centrality> tok_centSrc_;

  std::string d0IDName_;

//abby   
//then need to convert PAT candidates to reco candidates
  pat::CompositeCandidateCollection theGoodCandidates;
  MVACollection theMVANew;
  MVACollection theMVANew_xg;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

VertexCompositeSelector::VertexCompositeSelector(const edm::ParameterSet &iConfig, const ONNXRuntime *cache)
	:bdt("../../VertexCompositeAnalyzer/data/bdt_cuts.csv"), input_shapes_(), onnxRuntime_(cache)
{
	string a1 = "log3ddls";
	string a2 = "nVtxProb";
	string a3 = "n3DPointingAngle";
	string a4 = "nDtrk1Pt";
	string a5 = "nDtrk2Pt";
	string a6 = "nxyDCASigD1";
	string a7 = "nxyDCASigD2";
	string a8 = "nzDCASigD1";
	string a9 = "nzDCASigD2";
	string a10 = "npT";
	string a11 =  "ny";
	string a12 = "ncent";

  theInputVars.push_back(a1);
  theInputVars.push_back(a2);
  theInputVars.push_back(a3);
  theInputVars.push_back(a4);
  theInputVars.push_back(a5);
  theInputVars.push_back(a6);
  theInputVars.push_back(a7);
  theInputVars.push_back(a8);
  theInputVars.push_back(a9);
  theInputVars.push_back(a10);
  theInputVars.push_back(a12);
  theInputVars.push_back(a11);
  mva = new ReadBDT(theInputVars);
  // options

  PID_ = iConfig.getUntrackedParameter<int>("PID");
  PID_dau1_ = iConfig.getUntrackedParameter<int>("PID_dau1");
  PID_dau2_ = iConfig.getUntrackedParameter<int>("PID_dau2");

  // cut variables
  centMin_ = iConfig.getUntrackedParameter<int>("centMin", 0);
  centMax_ = iConfig.getUntrackedParameter<int>("centMax", 10000);
  multMin_ = iConfig.getUntrackedParameter<double>("multMin", -1);
  multMax_ = iConfig.getUntrackedParameter<double>("multMax", -1);
  deltaR_ = iConfig.getUntrackedParameter<double>("deltaR", 0.03);
  mvaMax_ = iConfig.getUntrackedParameter<double>("mvaMax", 999.9);
  mvaMin_ = iConfig.getUntrackedParameter<double>("mvaMin", -999.9);

  trkHighPurity_ = iConfig.getUntrackedParameter<bool>("trkHighPurity");
  trkPMin_ = iConfig.getUntrackedParameter<double>("trkPMin");
  trkPtMin_ = iConfig.getUntrackedParameter<double>("trkPtMin");
  trkEtaMax_ = iConfig.getUntrackedParameter<double>("trkEtaMax");
  trkPSumMin_ = iConfig.getUntrackedParameter<double>("trkPSumMin");
  trkPtSumMin_ = iConfig.getUntrackedParameter<double>("trkPtSumMin");
  trkPtAsymMin_ = iConfig.getUntrackedParameter<double>("trkPtAsymMin");
  trkEtaDiffMax_ = iConfig.getUntrackedParameter<double>("trkEtaDiffMax");
  trkPtErrMax_ = iConfig.getUntrackedParameter<double>("trkPtErrMax");
  trkNHitMin_ = iConfig.getUntrackedParameter<int>("trkNHitMin");
  candpTMin_ = iConfig.getUntrackedParameter<double>("candpTMin");
  candpTMax_ = iConfig.getUntrackedParameter<double>("candpTMax");
  candYMin_ = iConfig.getUntrackedParameter<double>("candYMin");
  candYMax_ = iConfig.getUntrackedParameter<double>("candYMax");
  cand3DDecayLengthSigMin_ = iConfig.getUntrackedParameter<double>("cand3DDecayLengthSigMin");
  cand2DDecayLengthSigMin_ = iConfig.getUntrackedParameter<double>("cand2DDecayLengthSigMin");
  cand3DPointingAngleMax_ = iConfig.getUntrackedParameter<double>("cand3DPointingAngleMax");
  cand2DPointingAngleMax_ = iConfig.getUntrackedParameter<double>("cand2DPointingAngleMax");
  cand3DDCAMin_ = iConfig.getUntrackedParameter<double>("cand3DDCAMin");
  cand3DDCAMax_ = iConfig.getUntrackedParameter<double>("cand3DDCAMax");
  cand2DDCAMin_ = iConfig.getUntrackedParameter<double>("cand2DDCAMin");
  cand2DDCAMax_ = iConfig.getUntrackedParameter<double>("cand2DDCAMax");
  candVtxProbMin_ = iConfig.getUntrackedParameter<double>("candVtxProbMin");
  mvaCut_ = iConfig.getParameter<double>(string("mvaCut"));


  // input tokens
  patCompositeCandidateCollection_Token_ = consumes<pat::CompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("D0")); 
  tok_offlinePV_ = consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("VertexCollection"));
  tok_generalTrk_ = consumes<reco::TrackCollection>(iConfig.getUntrackedParameter<edm::InputTag>("TrackCollection"));
  //abby
  Dedx_Token1_ = consumes<edm::ValueMap<reco::DeDxData>>(edm::InputTag("dedxHarmonic2"));
  Dedx_Token2_ = consumes<edm::ValueMap<reco::DeDxData>>(edm::InputTag("dedxTruncated40"));
  tok_genParticle_ = consumes<reco::GenParticleCollection>(edm::InputTag(iConfig.getUntrackedParameter<edm::InputTag>("GenParticleCollection")));

  usePID_ = false;
  selectFlavor_ = 0;
  if (iConfig.exists("usePID"))
    usePID_ = iConfig.getParameter<bool>("usePID");
  if (iConfig.exists("useFlavor"))
    selectFlavor_ = iConfig.getUntrackedParameter<int>("selectFlavor");

  // Loading TMVA
  useAnyMVA_ = false;




  isCentrality_ = false;
  if (iConfig.exists("isCentrality"))
    isCentrality_ = iConfig.getParameter<bool>("isCentrality");
  if (isCentrality_)
  {
    tok_centBinLabel_ = consumes<int>(iConfig.getParameter<edm::InputTag>("centralityBinLabel"));
    tok_centSrc_ = consumes<reco::Centrality>(iConfig.getParameter<edm::InputTag>("centralitySrc"));
  }

  if (iConfig.exists("useAnyMVA"))
    useAnyMVA_ = iConfig.getParameter<bool>("useAnyMVA");

	  if (useAnyMVA_) {
    	if (iConfig.exists("input_names")||iConfig.exists("output_names")) {
    	  input_names_ = iConfig.getParameter<std::vector<std::string>>("input_names");
    	  output_names_ = iConfig.getParameter<std::vector<std::string>>("output_names");
    	} else {
    	  throw cms::Exception("Configuration") << "onnxModelName not provided in ParameterSet";
    	}
	  }
 	


  d0IDName_ = (iConfig.getUntrackedParameter<edm::InputTag>("VertexCompositeCollection")).instance();

  produces<pat::CompositeCandidateCollection>(d0IDName_);
  produces<MVACollection>(Form("MVAValuesNew%s", d0IDName_.c_str()));

  //produces<pat::CompositeCandidateCollection>(d0IDName_);
  produces<MVACollection>(Form("MVAValuesNew%s2", d0IDName_.c_str()));

  isPionD1 = true;
  isPionD2 = true;
  isKaonD1 = false;
  isKaonD2 = false;
}
std::unique_ptr<ONNXRuntime> VertexCompositeSelector::initializeGlobalCache(const edm::ParameterSet &iConfig) {
   bool useAnyMVA = iConfig.exists("useAnyMVA") ? iConfig.getParameter<bool>("useAnyMVA") : false;

   if (!useAnyMVA) return nullptr;

   if (iConfig.exists("onnxModelFileName")) {
     std::string onnxModelPath = iConfig.getParameter<std::string>("onnxModelFileName");

     edm::FileInPath fip(Form("VertexCompositeAnalysis/VertexCompositeProducer/data/%s", onnxModelPath.c_str()));
     std::string fullPath = fip.fullPath();
     std::cout << fullPath << std::endl;

     std::ifstream testFile(fullPath);
     if (!testFile.good()) {
       throw cms::Exception("Configuration") << "cannot find ONNX Model in : " << fullPath;
     }
     testFile.close();

      return std::make_unique<ONNXRuntime>(fip.fullPath());

   }

   return nullptr;
}
void VertexCompositeSelector::globalEndJob(const ONNXRuntime *cache) {}

VertexCompositeSelector::~VertexCompositeSelector()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void VertexCompositeSelector::produce(edm::Event &iEvent, const edm::EventSetup &iSetup)
{

  using std::vector;
  using namespace edm;

  edm::Handle<pat::CompositeCandidateCollection> patCandidates;
  iEvent.getByToken(patCompositeCandidateCollection_Token_, patCandidates);

  if (!patCandidates.isValid())
  {
    edm::LogError("VertexCompositeSelector") << "Error: patCandidates collection not found!";
    return;
  }

  fillRECO(iEvent, iSetup);

  auto theNewD0Cands = std::make_unique<pat::CompositeCandidateCollection>();
  theNewD0Cands->reserve(theGoodCandidates.size());

  std::copy(theGoodCandidates.begin(),
            theGoodCandidates.end(),
            std::back_inserter(*theNewD0Cands));

  // Store final EDM output
  iEvent.put(std::move(theNewD0Cands), d0IDName_);


  theGoodCandidates.clear();

if (useAnyMVA_) {
    
    auto mvas = std::make_unique<MVACollection>(theMVANew.begin(), theMVANew.end());
    auto mvas_xg = std::make_unique<MVACollection>(theMVANew_xg.begin(), theMVANew_xg.end());
    
    iEvent.put(std::move(mvas), Form("MVAValuesNew%s", d0IDName_.c_str()));
    iEvent.put(std::move(mvas_xg), Form("MVAValuesNew%s2", d0IDName_.c_str()));
    
    theMVANew.clear();
    theMVANew_xg.clear();
}


}

void VertexCompositeSelector::fillRECO(edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  // get collections
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(tok_offlinePV_, vertices);

  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tok_generalTrk_, tracks);

  edm::Handle<pat::CompositeCandidateCollection> d0candidates;
  iEvent.getByToken(patCompositeCandidateCollection_Token_, d0candidates);
  const pat::CompositeCandidateCollection *d0candidates_ = d0candidates.product();

  edm::Handle<MVACollection> mvavalues;

  edm::Handle<edm::ValueMap<reco::DeDxData>> dEdxHandle1;
  if (usePID_)
    iEvent.getByToken(Dedx_Token1_, dEdxHandle1);

  edm::Handle<edm::ValueMap<reco::DeDxData>> dEdxHandle2;
  if (usePID_)
    iEvent.getByToken(Dedx_Token2_, dEdxHandle2);

  centrality = -1;
  if (isCentrality_)
  {
    edm::Handle<reco::Centrality> cent;
    iEvent.getByToken(tok_centSrc_, cent);

    iEvent.getByToken(tok_centBinLabel_, cbin_);
    centrality = *cbin_;
  }
  if (centrality != -1 && (centrality >= centMax_ || centrality < centMin_))
    return;

  // best vertex
  bestvz = -999.9;
  bestvx = -999.9;
  bestvy = -999.9;
  double bestvzError = -999.9, bestvxError = -999.9, bestvyError = -999.9;
  const reco::Vertex &vtx = (*vertices)[0];
  bestvz = vtx.z();
  bestvx = vtx.x();
  bestvy = vtx.y();
  bestvzError = vtx.zError();
  bestvxError = vtx.xError();
  bestvyError = vtx.yError();

  // Ntrkoffline
  Ntrkoffline = 0;
  if (multMax_ != -1 && multMin_ != -1)
  {
    for (unsigned it = 0; it < tracks->size(); ++it)
    {

      const reco::Track &trk = (*tracks)[it];

      math::XYZPoint bestvtx(bestvx, bestvy, bestvz);

      double dzvtx = trk.dz(bestvtx);
      double dxyvtx = trk.dxy(bestvtx);
      double dzerror = sqrt(trk.dzError() * trk.dzError() + bestvzError * bestvzError);
      double dxyerror = sqrt(trk.d0Error() * trk.d0Error() + bestvxError * bestvyError);

      if (!trk.quality(reco::TrackBase::highPurity))
        continue;
      if (fabs(trk.ptError()) / trk.pt() > 0.10)
        continue;
      if (fabs(dzvtx / dzerror) > 3)
        continue;
      if (fabs(dxyvtx / dxyerror) > 3)
        continue;

      double eta = trk.eta();
      double pt = trk.pt();

      if (fabs(eta) > 2.4)
        continue;
      if (pt <= 0.4)
        continue;
      Ntrkoffline++;
    }

    if (Ntrkoffline >= multMax_ || Ntrkoffline < multMin_)
      return;
  }


  // RECO Candidate info
  //cout << "d0candidates->size() = " << recoCandidates.size() << endl;
  for(unsigned it = 0; it < d0candidates_->size(); ++it)
  //for(unsigned it = 0; it < 10; ++it)
  {

    //cout << "*** it = " << it << " **" << endl;
    const pat::CompositeCandidate &trk = (*d0candidates_)[it];

    double bdt_cut_value = -999.9;
    double secvz = -999.9, secvx = -999.9, secvy = -999.9;
    secvz = trk.userFloat("vertexZ");
    secvx = trk.userFloat("vertexX");
    secvy = trk.userFloat("vertexY");

    eta = trk.eta();
    y = trk.rapidity();
    pt = trk.pt();
    flavor = trk.pdgId() / 421;

    double px = trk.px();
    double py = trk.py();
    double pz = trk.pz();
    mass = trk.mass();

    const reco::Candidate *d1 = trk.daughter(0);
    const reco::Candidate *d2 = trk.daughter(1);

    // Gen match
    //  select particle vs antiparticle
    if (usePID_ && selectFlavor_ && (int)flavor != selectFlavor_)
      continue;
    //cout << "cut 1 " << endl;

    // select on pT and y
    if (pt < candpTMin_ || pt > candpTMax_)
      continue;
    //cout << "cut 2 " << endl;
    if (y < candYMin_ || y > candYMax_)
      continue;
    //cout << "cut 3 " << endl;
    // pt
    pt1 = d1->pt();
    pt2 = d2->pt();

    if (pt1 < trkPtMin_ || pt2 < trkPtMin_)
      continue;
    //cout << "cut 4 " << endl;
    if ((pt1 + pt2) < trkPtSumMin_)
      continue;
    //cout << "cut 5 " << endl;

    if (pt2 / pt1 < trkPtAsymMin_ || pt1 / pt2 < trkPtAsymMin_)
      continue;
    //cout << "cut 6 " << endl;

    // momentum
    p1 = d1->p();
    p2 = d2->p();

    if (p1 < trkPMin_ || p2 < trkPMin_)
      continue;
    //cout << "cut 7 " << endl;
    if ((p1 + p2) < trkPSumMin_)
      continue;
    //cout << "cut 8 " << endl;

    // eta
    eta1 = d1->eta();
    eta2 = d2->eta();

    if (fabs(eta1) > trkEtaMax_ || fabs(eta2) > trkEtaMax_)
      continue;
    //cout << "cut 9 " << endl;
    if (fabs(eta1 - eta2) > trkEtaDiffMax_)
      continue;
    //cout << "cut 10 " << endl;

    // phi
    phi1 = d1->phi();
    phi2 = d2->phi();

    // vtxChi2
    vtxChi2 = trk.userFloat("vertexChi2");
    ndf = trk.userFloat("vertexNdof");
    VtxProb = TMath::Prob(vtxChi2, ndf);

    if (VtxProb < candVtxProbMin_)
      continue;
    //cout << "cut 11 " << endl;

    // PAngle
    TVector3 ptosvec(secvx - bestvx, secvy - bestvy, secvz - bestvz);
    TVector3 secvec(px, py, pz);


    //cout << "bestvx = " << bestvx << " bestvy = " << bestvy << " bestvz = " << bestvz << endl;
    //cout << "secvx = " << secvx << " secvy = " << secvy << " secvz = " << secvz << endl;
    //cout << "px = " << px << " py = " << py << " pz = " << pz << endl;


    TVector3 ptosvec2D(secvx - bestvx, secvy - bestvy, 0);
    TVector3 secvec2D(px, py, 0);

    agl = cos(secvec.Angle(ptosvec));
    agl_abs = secvec.Angle(ptosvec);
    //cout << "agl_abs = " << agl_abs << " and cut = " << cand3DPointingAngleMax_ << endl;
    if (agl_abs > cand3DPointingAngleMax_)
      continue;
    //cout << "cut 12 " << endl;

    agl2D = cos(secvec2D.Angle(ptosvec2D));
    agl2D_abs = secvec2D.Angle(ptosvec2D);
    if (agl2D_abs > cand2DPointingAngleMax_)
      continue;
    //cout << "cut 13 " << endl;

    // Decay length 3D
    typedef ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>> SMatrixSym3D;
    typedef ROOT::Math::SVector<double, 3> SVector3;
    typedef ROOT::Math::SVector<double, 6> SVector6;

    reco::Vertex::CovarianceMatrix sec_covariance;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				sec_covariance(i, j) = trk.userFloat("vertexCovariance_" + std::to_string(i) + "_" + std::to_string(j));
			}
		}

    SMatrixSym3D totalCov = vtx.covariance() + sec_covariance;
    SVector3 distanceVector(secvx - bestvx, secvy - bestvy, secvz - bestvz);

    dl = ROOT::Math::Mag(distanceVector);
    dlerror = sqrt(ROOT::Math::Similarity(totalCov, distanceVector)) / dl;

    dlos = dl / dlerror;

    //cout << "dl = " << dl << " dlerror = " << dlerror << endl;
    //cout << "dlos = " << dlos << endl;
    if (dlos < cand3DDecayLengthSigMin_ || dlos > 1000.)
      continue;
    //cout << "cut 14 " << endl;

    // Decay length 2D
    SVector6 v1(vtx.covariance(0, 0), vtx.covariance(0, 1), vtx.covariance(1, 1), 0, 0, 0);
    SVector6 v2(sec_covariance(0, 0), sec_covariance(0, 1), sec_covariance(1, 1), 0, 0, 0);

    SMatrixSym3D sv1(v1);
    SMatrixSym3D sv2(v2);

    SMatrixSym3D totalCov2D = sv1 + sv2;
    SVector3 distanceVector2D(secvx - bestvx, secvy - bestvy, 0);

    double dl2D = ROOT::Math::Mag(distanceVector2D);
    double dl2Derror = sqrt(ROOT::Math::Similarity(totalCov2D, distanceVector2D)) / dl2D;

    dlos2D = dl2D / dl2Derror;
    if (dlos2D < cand2DDecayLengthSigMin_ || dlos2D > 1000.)
      continue;
    //cout << "cut 15 " << endl;

    double dca3D = dl * sin(agl_abs);
    if (dca3D < cand3DDCAMin_ || dca3D > cand3DDCAMax_)
      continue;
    //cout << "cut 16 " << endl;

    double dca2D = dl2D * sin(agl2D_abs);
    if (dca2D < cand2DDCAMin_ || dca2D > cand2DDCAMax_)
      continue;
    //cout << "cut 17 " << endl;

    // trk info
    auto dau1 = d1->get<reco::TrackRef>();
    // trk quality
    trkquality1 = dau1->quality(reco::TrackBase::highPurity);
    if (trkHighPurity_ && !trkquality1)
      continue;
    //cout << "cut 18 " << endl;

    // trk dEdx
    H2dedx1 = -999.9;
    T4dedx1 = -999.9;
    if (usePID_)
    {
      if (dEdxHandle1.isValid())
      {
        const edm::ValueMap<reco::DeDxData> dEdxTrack = *dEdxHandle1.product();
        H2dedx1 = dEdxTrack[dau1].dEdx();
        if (H2dedx1 > (2.8 / pow(pt1 * cosh(eta1), 0.4) + 0.2) && H2dedx1 < (2.8 / pow(pt1 * cosh(eta1), 0.9) + 1.8) && H2dedx1 > (2.8 / pow(0.75, 0.4) + 0.2))
        {
          isKaonD1 = true;
          isPionD1 = false;
        }
        if ((H2dedx1 < (2.8 / pow(pt1 * cosh(eta1), 0.4) + 0.2) || H2dedx1 < (2.8 / pow(0.75, 0.4) + 0.2)) && H2dedx1 > 0)
        {
          isPionD1 = true;
          isKaonD1 = false;
        }
      }

      if (dEdxHandle2.isValid())
      {
        const edm::ValueMap<reco::DeDxData> dEdxTrack = *dEdxHandle2.product();
        T4dedx1 = dEdxTrack[dau1].dEdx();
      }
    }

    // track Chi2
    trkChi1 = dau1->normalizedChi2();

    // track pT error
    ptErr1 = dau1->ptError();
    if (ptErr1 / dau1->pt() > trkPtErrMax_)
      continue;
    //cout << "cut 19 " << endl;

    /*
    secvz = trk.vz();
    secvx = trk.vx();
    secvy = trk.vy();
    */

    // trkNHits
    nhit1 = dau1->numberOfValidHits();
    if (nhit1 < trkNHitMin_)
      continue;
    //cout << "cut 20 " << endl;
    // DCA
    math::XYZPoint bestvtx(bestvx, bestvy, bestvz);

    double dzbest1 = dau1->dz(bestvtx);
    double dxybest1 = dau1->dxy(bestvtx);
    double dzerror1 = sqrt(dau1->dzError() * dau1->dzError() + bestvzError * bestvzError);
    double dxyerror1 = sqrt(dau1->d0Error() * dau1->d0Error() + bestvxError * bestvyError);

    dzos1 = dzbest1 / dzerror1;
    dxyos1 = dxybest1 / dxyerror1;

    auto dau2 = d2->get<reco::TrackRef>();

    trkquality2 = dau2->quality(reco::TrackBase::highPurity);
    if (trkHighPurity_ && !trkquality2)
      continue;
    //cout << "cut 21 " << endl;

    // trk dEdx
    H2dedx2 = -999.9;
    T4dedx2 = -999.9;

    if (usePID_)
    {
      if (dEdxHandle1.isValid())
      {
        const edm::ValueMap<reco::DeDxData> dEdxTrack = *dEdxHandle1.product();
        H2dedx2 = dEdxTrack[dau2].dEdx();

        if (H2dedx2 > (2.8 / pow(pt2 * cosh(eta2), 0.4) + 0.2) && H2dedx2 < (2.8 / pow(pt2 * cosh(eta2), 0.9) + 1.8) && H2dedx2 > (2.8 / pow(0.75, 0.4) + 0.2))
        {
          isKaonD2 = true;
          isPionD2 = false;
        }
        if ((H2dedx2 < (2.8 / pow(pt2 * cosh(eta2), 0.4) + 0.2) || H2dedx2 < (2.8 / pow(0.75, 0.4) + 0.2)) && H2dedx2 > 0)
        {
          isPionD2 = true;
          isKaonD2 = false;
        }
      }

      if (dEdxHandle2.isValid())
      {
        const edm::ValueMap<reco::DeDxData> dEdxTrack = *dEdxHandle2.product();
        T4dedx2 = dEdxTrack[dau2].dEdx();
      }

      if (flavor > 0 && (!isPionD1 || !isKaonD2))
        continue;
      if (flavor < 0 && (!isPionD2 || !isKaonD1))
        continue;
    }

    // track Chi2
    trkChi2 = dau2->normalizedChi2();

    // track pT error
    ptErr2 = dau2->ptError();
    if (ptErr2 / dau2->pt() > trkPtErrMax_)
      continue;
    //cout << "cut 22 " << endl;

    // vertexCovariance 00-xError 11-y 22-z
    /*
    secvz = trk.vz();
    secvx = trk.vx();
    secvy = trk.vy();
    */

    // trkNHits
    nhit2 = dau2->numberOfValidHits();
    if (nhit2 < trkNHitMin_)
      continue;
    //cout << "cut 23 " << endl;

    // DCA
    // math::XYZPoint bestvtx(bestvx,bestvy,bestvz);

    double dzbest2 = dau2->dz(bestvtx);
    double dxybest2 = dau2->dxy(bestvtx);
    double dzerror2 = sqrt(dau2->dzError() * dau2->dzError() + bestvzError * bestvzError);
    double dxyerror2 = sqrt(dau2->d0Error() * dau2->d0Error() + bestvxError * bestvyError);

    dzos2 = dzbest2 / dzerror2;
    dxyos2 = dxybest2 / dxyerror2;


    mva_value = -999.9;
    if (useAnyMVA_ && onnxRuntime_)
    {
    //cout << "*** it = " << it << " **" << endl;

		 cms::Ort::FloatArrays data_;
         data_.emplace_back(19, 0);
         std::vector<float> &onnxVals_=data_[0];
				 onnxVals_[0] = pt;
				 onnxVals_[1] = y;
				 onnxVals_[2] = vtxChi2;
				 onnxVals_[3] = centrality;
				 onnxVals_[4] = agl;
				 onnxVals_[5] = agl_abs;
				 onnxVals_[6] = agl2D;
				 onnxVals_[7] = agl2D_abs;
				 onnxVals_[8] = dl;
				 onnxVals_[9] = dlos;
				 onnxVals_[10] = dl2D;
				 onnxVals_[11] = dlos2D;
				 onnxVals_[12] = pt1;
				 onnxVals_[13] = eta1;
				 onnxVals_[14] = pt2;
				 onnxVals_[15] = eta2;
				 onnxVals_[16] = ptErr1;
				 onnxVals_[17] = ptErr2;
				 onnxVals_[18] = trk.userFloat("track3DDCA");

				 std::vector<float> outputs = onnxRuntime_->run(input_names_, data_, input_shapes_,output_names_)[0];
				 float onnxVal = outputs[1];

			inputValues.clear();
			inputValues.push_back(log10(dlos)); // 00
			inputValues.push_back(VtxProb);     // 01
			inputValues.push_back(agl_abs); // 02
			inputValues.push_back(pt1);     // 03
			inputValues.push_back(pt2);     // 04
			inputValues.push_back(dxyos1);  // 04
			inputValues.push_back(dxyos2);  // 04
			inputValues.push_back(dzos1);   // 04
			inputValues.push_back(dzos2);
			inputValues.push_back(pt);
			inputValues.push_back(centrality);
			inputValues.push_back(y);
			mva_value = mva->GetMvaValue(inputValues);
			bdt_cut_value = bdt.getBDTCut(y, centrality, pt);
			if (mva_value <= bdt_cut_value || onnxVal <= mvaCut_) continue;
			/*
			   cout << "---------------------------" << endl;
			   cout << "y && cent && pt = " << y << " && " << centrality << " && " << pt << endl;
			   cout << "bdt_weight = " << mva_value << endl;
			   cout << "bdt_cut_value = " << bdt_cut_value << endl;
			   */
			//if (bdt_cut_value < -1) continue;

			theMVANew.push_back(mva_value);
			theMVANew_xg.push_back(onnxVal);
		//	mvaVals_.push_back(onnxVal);


      // if (mva_value > -900) cout << "BDT weight = " << mva_value << endl;
    }



		// select MVA value
		theGoodCandidates.push_back(trk);
	}
}

// ------------ method called once each job just before starting event
// loop  ------------
void VertexCompositeSelector::beginJob()
{
}

// ------------ method called once each job just after ending the event
// loop  ------------
void VertexCompositeSelector::endJob()
{
}

// define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
DEFINE_FWK_MODULE(VertexCompositeSelector);
