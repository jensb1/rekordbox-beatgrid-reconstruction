#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mach-o/dyld.h>
#include <vector>
#include <cmath>

struct BeatInfo { int pos; int pad; float interval; int numBar; int x; int flag; }; // 24 bytes
typedef void (*Ctor)(void*);
typedef int (*Create)(void*, int, int);
typedef void (*PutSample)(void*, short*, int);
typedef void (*BpmAdjust)(void*, float*, std::vector<BeatInfo>*, int);
typedef void (*BeatPosAdjust)(void*, std::vector<BeatInfo>*);
typedef void (*CBD)(void*, double, int);

int main(int argc,char**argv){
  if(argc<3){fprintf(stderr,"usage: %s audio.f32 beats.bin\n",argv[0]);return 1;}
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL); if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char* nm=_dyld_get_image_name(i); if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  Ctor ctor=(Ctor)(base+(0x100f02b3cULL-0x100000000ULL));
  Create create=(Create)(base+(0x100f02da0ULL-0x100000000ULL));
  PutSample put=(PutSample)(base+(0x100f0303cULL-0x100000000ULL));
  BpmAdjust badj=(BpmAdjust)(base+(0x100f032dcULL-0x100000000ULL));
  BeatPosAdjust padj=(BeatPosAdjust)(base+(0x100f03ab8ULL-0x100000000ULL));
  CBD cbd=(CBD)(base+(0x100f04cbcULL-0x100000000ULL));
  char ua[0x200]; memset(ua,0,sizeof(ua)); ctor(ua);
  FILE* fa=fopen(argv[1],"rb"); fseek(fa,0,SEEK_END); long asz=ftell(fa); fseek(fa,0,SEEK_SET);
  float* af=(float*)malloc(asz); fread(af,1,asz,fa); fclose(fa); int nf=asz/4;
  int ok=create(ua,44100,nf);
  fprintf(stderr,"create=%d fields +8=%d +c=%d +1c=%d\n",ok,*(int*)(ua+8),*(int*)(ua+0xc),*(int*)(ua+0x1c));
  short* st=(short*)malloc(nf*2*sizeof(short));
  for(int i=0;i<nf;i++){float v=af[i]; if(v>1)v=1; if(v<-1)v=-1; short s=(short)lrintf(v*32767.0f); st[2*i]=s; st[2*i+1]=s;}
  put(ua,st,nf);
  { short* lb=*(short**)(ua+0x78); int len=*(int*)(ua+0x1c); FILE* fl=fopen("/tmp/lowband.bin","wb"); fwrite(lb,2,len,fl); fclose(fl); fprintf(stderr,"lowband len=%d\n",len); }
  // read beats: 12-byte BA_BeatInfo (pos in 22050 units, numBar, bpm*100)
  FILE* fb=fopen(argv[2],"rb"); fseek(fb,0,SEEK_END); long bsz=ftell(fb); fseek(fb,0,SEEK_SET);
  int nb=bsz/12; int* raw=(int*)malloc(bsz); fread(raw,1,bsz,fb); fclose(fb);
  std::vector<BeatInfo> v;
  for(int i=0;i<nb;i++){
    int pos=raw[3*i], numBar=raw[3*i+1], bpm=raw[3*i+2];
    BeatInfo b; memset(&b,0,sizeof(b));
    b.pos=pos; b.interval=6000000.0f/bpm; b.numBar=numBar; b.x=-1; b.flag=0;
    v.push_back(b);
  }
  *(int*)(ua+0x20)=nb;
  float bpm=atof(argv[3]);
  fprintf(stderr,"input bpm=%.3f n=%d first pos=%d interval=%.3f\n",bpm,nb,v[0].pos,v[0].interval);
  badj(ua,&bpm,&v,nb);
  fprintf(stderr,"after bpmAdjust bpm=%.9f n=%zu (bits %08x)\n",bpm,v.size(),*(unsigned*)&bpm);
  {
    struct IVec { int* b; int* e; int* c; };
    IVec* ob=*(IVec**)(ua+0x58); IVec* oe=*(IVec**)(ua+0x60);
    fprintf(stderr,"candidate grids: %ld\n",(long)(oe-ob));
    int minfirst=0x7fffffff,maxfirst=-0x7fffffff,minstep=0x7fffffff,maxstep=-0x7fffffff;
    for(IVec* it=ob; it<oe; ++it){
      long n=it->e-it->b; if(n<2) continue;
      int f=it->b[0], st=it->b[1]-it->b[0];
      if(f<minfirst)minfirst=f; if(f>maxfirst)maxfirst=f;
      if(st<minstep)minstep=st; if(st>maxstep)maxstep=st;
    }
    fprintf(stderr,"candidate first range %d..%d step range %d..%d\n",minfirst,maxfirst,minstep,maxstep);
    FILE* fc=fopen("/tmp/candidates.bin","wb");
    int ncand=(int)(oe-ob); fwrite(&ncand,4,1,fc);
    for(IVec* it=ob; it<oe; ++it){int n=(int)(it->e-it->b); fwrite(&n,4,1,fc); fwrite(it->b,4,n,fc);} fclose(fc);
    fprintf(stderr,"wrote candidates %d\n",ncand);
  }
  for(size_t i=0;i<v.size() && i<5;i++) fprintf(stderr,"  %zu pos=%d int=%.3f bar=%d x=%d flag=%d\n",i,v[i].pos,v[i].interval,v[i].numBar,v[i].x,v[i].flag);
  size_t n1=v.size();
  padj(ua,&v);
  if(argc>4){FILE* fo=fopen(argv[4],"wb"); fwrite(v.data(),sizeof(BeatInfo),v.size(),fo); fclose(fo);}
  fprintf(stderr,"ua +1c=%d +20=%d\n",*(int*)(ua+0x1c),*(int*)(ua+0x20));
  if(argc>5){int start=atoi(argv[5]); cbd(ua,bpm,start); int* b=*(int**)(ua+0x58); int* e=*(int**)(ua+0x60); fprintf(stderr,"grid start=%d n=%ld first=%d last=%d\n",start,(long)(e-b),b[0],e[-1]);}
  fprintf(stderr,"after beatPositionAdjust n=%zu\n",v.size());
  for(size_t i=0;i<v.size() && i<5;i++) fprintf(stderr,"  %zu pos=%d int=%.3f bar=%d x=%d flag=%d\n",i,v[i].pos,v[i].interval,v[i].numBar,v[i].x,v[i].flag);
  if(v.size()>n1) { for(size_t i=n1;i<v.size() && i<n1+5;i++) fprintf(stderr,"  new %zu pos=%d int=%.3f bar=%d\n",i,v[i].pos,v[i].interval,v[i].numBar); }
  return 0;
}
