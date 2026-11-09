#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
#include <initializer_list>
typedef void (*Ctor)(void*, int);
typedef int (*CreateFn)(void*, bool, unsigned long, int, int, bool, bool, bool);
typedef int (*SetSampleFn)(void*, float*, float*, int);
struct BA { int pos; int pad; float iv; int a; int b; unsigned char flag,c1,c2,c3; };
int main(int argc,char**argv){
  void* h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  FILE* fa=fopen(argv[2],"rb"); fseek(fa,0,SEEK_END); long asz=ftell(fa); fseek(fa,0,SEEK_SET);
  float* a=(float*)malloc(asz); fread(a,1,asz,fa); fclose(fa); int ns=asz/4;
  char w[0x100]; memset(w,0,sizeof(w));
  ((Ctor)(base+(0x101b766b4ULL-0x100000000ULL)))(w,2);
  int cok=((CreateFn)(base+(0x101b766f0ULL-0x100000000ULL)))(w,1,44100,ns,2,1,1,0);
  int sok=((SetSampleFn)(base+(0x101b76850ULL-0x100000000ULL)))(w,a,a,ns);
  void* an=*(void**)(w+8);
  fprintf(stderr,"create=%d setsample=%d\n",cok,sok);
  for(int off : {0x58, 0xe8, 0x100, 0x2c0}) {
    BA* bb=*(BA**)((char*)an+off); BA* ee=*(BA**)((char*)an+off+8);
    long nn=(char*)ee-(char*)bb;
    fprintf(stderr,"vector at an+0x%x: count=%ld\n",off,(long)(nn/24));
    if(nn>0 && nn<20000000 && nn%24==0){
      for(long i=0;i<nn/24 && i<5;i++)
        fprintf(stderr,"   %ld: pos=%d iv=%.4f a=%d b=%d f=%d\n",i,bb[i].pos,bb[i].iv,bb[i].a,bb[i].b,bb[i].flag);
    }
  }
  BA* b=*(BA**)((char*)an+0x58);
  BA* e=*(BA**)((char*)an+0x60);
  long n=(e-b);
  fprintf(stderr,"vector at +0x58: %p .. %p  count=%ld\n",(void*)b,(void*)e,n);
  if(n>0 && n<100000){
    for(long i=0;i<n && i<8;i++)
      fprintf(stderr,"  %ld: pos=%d iv=%.4f a=%d b=%d f=%d\n",i,b[i].pos,b[i].iv,b[i].a,b[i].b,b[i].flag);
    FILE* fo=fopen("/tmp/peaks.bin","wb"); fwrite(b,sizeof(BA),n,fo); fclose(fo);
    fprintf(stderr,"wrote /tmp/peaks.bin\n");
  }
  return 0;
}
