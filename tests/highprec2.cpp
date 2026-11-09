#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <mach-o/dyld.h>
#include "json.hpp"

typedef void (*StrCtor)(void* self, const char* text);
typedef bool (*AnalyzeFn)(void* model, void* unused, const std::string* path, int a, int b, bool flag, nlohmann::ordered_json* out);
typedef int (*CreateFn)(void*, int, unsigned long, int, int, int, int, int);
typedef int (*SetSampleFn)(void*, float*, float*, int);
typedef int (*FocusFn)(void*, void*, int, int, void*, int, int);
typedef int (*GetIntFn)(void*);
typedef int (*GetPosFn)(void*, void*);
struct BA { int pos; int numBar; int bpm; };

int main(int argc,char**argv){
  if(argc<4){fprintf(stderr,"usage: %s rb.dylib audio.m4a audio.f32\n",argv[0]);return 1;}
  void* h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL); if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char* nm=_dyld_get_image_name(i); if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  StrCtor ctor=(StrCtor)(base+(0x102037cf4ULL-0x100000000ULL));
  AnalyzeFn analyze=(AnalyzeFn)(base+(0x101ebac74ULL-0x100000000ULL));
  char mbuf[64]={0}; ctor(mbuf,"/Applications/rekordbox 7/rekordbox.app/Contents/Resources/models/detect_beat/model8");
  char ubuf[8]={0}; ctor(ubuf,"");
  std::string path=argv[2];
  nlohmann::ordered_json out;
  bool ok=analyze(mbuf,ubuf,&path,44100,1,1,&out);
  if(!ok){fprintf(stderr,"analyze failed\n");return 1;}
  int topbpm=argc>4?atoi(argv[4]):out["bpm"].get<int>();
  auto& bl=out["beatList"];
  int nb=(int)bl.size();
  fprintf(stderr,"ok bpm=%d beats=%d\n",topbpm,nb);
  std::vector<BA> ba(nb);
  { const char* bf=getenv("BEATS_FILE");
    if(bf){ FILE* fb=fopen(bf,"rb"); fseek(fb,0,SEEK_END); long bsz=ftell(fb); fseek(fb,0,SEEK_SET);
      int n2=bsz/sizeof(BA); ba.resize(n2); fread(ba.data(),sizeof(BA),n2,fb); fclose(fb);
      nb=n2; fprintf(stderr,"beats from %s: nb=%d\n",bf,nb);
    } else
    for(int i=0;i<nb;i++){ba[i].pos=bl[i]["pos"].get<int>(); ba[i].numBar=bl[i]["numBar"].get<int>(); ba[i].bpm=bl[i]["bpm"].get<int>();} }
  // load audio f32
  FILE* fa=fopen(argv[3],"rb"); fseek(fa,0,SEEK_END); long asz=ftell(fa); fseek(fa,0,SEEK_SET);
  float* audio=(float*)malloc(asz); fread(audio,1,asz,fa); fclose(fa); int ns=asz/4;
  unsigned char wrapper[0x18]; memset(wrapper,0,sizeof(wrapper));
  ((void(*)(void*,int))(base+(0x101b766b4ULL-0x100000000ULL)))(wrapper,2);
  int cok=((CreateFn)(base+(0x101b766f0ULL-0x100000000ULL)))(wrapper,1,44100,ns,32,1,1,0);
  fprintf(stderr,"create=%d\n",cok);
  int sok=((SetSampleFn)(base+(0x101b76850ULL-0x100000000ULL)))(wrapper,audio,audio,ns);
  fprintf(stderr,"setsample=%d\n",sok);
  int off_before=((GetIntFn)(base+(0x101b76a34ULL-0x100000000ULL)))(wrapper);
  fprintf(stderr,"GetOffsetCalc before focus=%d\n",off_before);
  int fok=((FocusFn)(base+(0x101b76a9cULL-0x100000000ULL)))(wrapper,NULL,nb,topbpm,ba.data(),nb,4);
  fprintf(stderr,"focus=%d\n",fok);
  int offset=((GetIntFn)(base+(0x101b76a34ULL-0x100000000ULL)))(wrapper);
  fprintf(stderr,"GetOffsetCalc=%d\n",offset);
  void* analyzer=*(void**)(wrapper+8);
  for(int o=0;o<0x400;o+=4){float f=*(float*)((char*)analyzer+o); if(f>100 && f<150) fprintf(stderr,"analyzer +%x = %.9f\n",o,f);}
  { char* ua=(char*)analyzer+0x650; for(int o=0;o<0x100;o+=4){float f=*(float*)(ua+o); if(f>100 && f<150) fprintf(stderr,"ua +%x = %.9f\n",o,f);} }
  int obpm=((GetIntFn)(base+(0x101b76ae0ULL-0x100000000ULL)))(wrapper);
  int onum=((GetIntFn)(base+(0x101b76b04ULL-0x100000000ULL)))(wrapper);
  fprintf(stderr,"out bpm=%d num=%d\n",obpm,onum);
  BA* res=(BA*)calloc(onum,sizeof(BA));
  ((GetPosFn)(base+(0x101b76b28ULL-0x100000000ULL)))(wrapper,res);
  FILE* fo=fopen("/tmp/highprec_out.bin","wb"); fwrite(res,sizeof(BA),onum,fo); fclose(fo);
  for(int i=0;i<onum && i<8;i++) fprintf(stderr,"%d: pos=%d bar=%d bpm=%d\n",i,res[i].pos,res[i].numBar,res[i].bpm);
  return 0;
}
