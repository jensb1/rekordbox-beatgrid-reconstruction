#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
typedef void (*Ctor)(void*, int);
typedef int (*CreateFn)(void*, bool, unsigned long, int, int, bool, bool, bool);
typedef int (*SetSampleFn)(void*, float*, float*, int);
typedef void (*FocusFn)(void*, void*, int, int, void*, int, int);
typedef int (*GetIntFn)(void*);
typedef void (*GetPosFn)(void*, void*);
struct BA { int pos; int numBar; int bpm; };
int main(int argc,char**argv){
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  int x = argc>1?atoi(argv[1]):32;
  int bpm = argc>2?atoi(argv[2]):12200;
  FILE* fa=fopen("/tmp/h.f32","rb"); fseek(fa,0,SEEK_END); long asz=ftell(fa); fseek(fa,0,SEEK_SET);
  float* a=(float*)malloc(asz); fread(a,1,asz,fa); fclose(fa); int ns=asz/4;
  char w[0x100]; memset(w,0,sizeof(w));
  ((Ctor)(base+(0x101b766b4ULL-0x100000000ULL)))(w,2);
  int cok=((CreateFn)(base+(0x101b766f0ULL-0x100000000ULL)))(w,1,44100,ns,x,1,1,0);
  int sok=((SetSampleFn)(base+(0x101b76850ULL-0x100000000ULL)))(w,a,a,ns);
  FILE* fb=fopen(argc>3?argv[3]:"/tmp/h_beats_44.bin","rb");
  fseek(fb,0,SEEK_END); long bsz=ftell(fb); fseek(fb,0,SEEK_SET);
  int nb=bsz/12; BA* in=(BA*)calloc(nb+8,sizeof(BA)); fread(in,1,bsz,fb); fclose(fb);
  BA out[4000]; memset(out,0,sizeof(out));
  ((FocusFn)(base+(0x101b76a9cULL-0x100000000ULL)))(w,in,nb,bpm,out,nb,4);
  int onum=((GetIntFn)(base+(0x101b76b04ULL-0x100000000ULL)))(w);
  BA* res=(BA*)calloc(onum>0?onum:1,sizeof(BA));
  ((GetPosFn)(base+(0x101b76b28ULL-0x100000000ULL)))(w,res);
  printf("nb=%d x=%d bpm=%d create=%d setsample=%d num=%d  first=%d %d %d\n",x,bpm,nb,cok,sok,onum,
         onum>0?res[0].pos:-1, onum>1?res[1].pos:-1, onum>2?res[2].pos:-1);
  return 0;
}
