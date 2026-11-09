#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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
  FILE* fa=fopen(argv[1],"rb"); fseek(fa,0,SEEK_END); long asz=ftell(fa); fseek(fa,0,SEEK_SET);
  float* a=(float*)malloc(asz); fread(a,1,asz,fa); fclose(fa); int ns=asz/4;
  char ua[0x400]; memset(ua,0,sizeof(ua));
  ((Ctor)(base+(0x100f02b3cULL-0x100000000ULL)))(ua);
  int ok=((Create)(base+(0x100f02da0ULL-0x100000000ULL)))(ua,44100,ns);
  short* st=(short*)malloc((size_t)ns*2*sizeof(short));
  for(int i=0;i<ns;i++){
    float v=a[i]; if(v>1)v=1; if(v<-1)v=-1;
    short s=(short)lrintf(v*32767.0f); st[2*i]=s; st[2*i+1]=s;
  }
  int chunk=400000;
  for(int pos=0;pos<ns;pos+=chunk){
    int n=ns-pos; if(n>chunk) n=chunk;
    ((Put)(base+(0x100f0303cULL-0x100000000ULL)))(ua,st+(size_t)pos*2,n);
  }
  fprintf(stderr,"create=%d ns=%d\n",ok,ns);
  const char* nm3[3]={"/tmp/env70.bin","/tmp/env78.bin","/tmp/env80.bin"};
  int offs[3]={0x70,0x78,0x80};
  int len=*(int*)(ua+0x1c);
  for(int k=0;k<3;k++){
    short* p=*(short**)(ua+offs[k]);
    FILE* f=fopen(nm3[k],"wb"); fwrite(p,2,len,f); fclose(f);
    fprintf(stderr,"%s: len=%d\n",nm3[k],len);
  }
  return 0;
}
