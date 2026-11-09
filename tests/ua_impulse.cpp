#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
typedef void (*Ctor)(void*);
typedef int (*Create)(void*, int, int);
typedef void (*Put)(void*, short*, int);
int main(int argc,char**argv){
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  int n=8192;
  char ua[0x400]; memset(ua,0,sizeof(ua));
  ((Ctor)(base+(0x100f02b3cULL-0x100000000ULL)))(ua);
  int ok=((Create)(base+(0x100f02da0ULL-0x100000000ULL)))(ua,44100,n);
  short* st=(short*)calloc(n*2,sizeof(short));
  st[0]=30000; st[1]=30000;                       // impulse at frame 0
  ((Put)(base+(0x100f0303cULL-0x100000000ULL)))(ua,st,n);
  fprintf(stderr,"create=%d\n",ok);
  const char* names[3]={"+0x70","+0x78","+0x80"};
  int offs[3]={0x70,0x78,0x80};
  for(int k=0;k<3;k++){
    short* p=*(short**)(ua+offs[k]);
    printf("%s -> %p :", names[k], (void*)p);
    for(int i=0;i<24;i++) printf(" %d", p?p[i]:-1);
    printf("\n");
  }
  return 0;
}
