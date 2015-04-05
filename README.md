# RuntimeCompiledCudafy
A PoC showing that you can build Cudafy functions at runtime thus allowing for GPU acceleration for user-defined functions.

# Usage
The repo contains a simple example which uses CodeDOM to describe a Cudafy function which is then assembled and executed on the GPU at runtime.

Given you have the appropriate libraries and hardware installed, the program should compile and run without any major hassle whatsoever.

For reference this is the code that is generated from the CodeDOM:
```C#
namespace DynamicGPU {
    using System;
    using Cudafy;
    using Cudafy.Host;
    using Cudafy.Translator;
    
    
    public class DualMarchingCubes {
        
        [Cudafy()]
        public static void ScalarField(Cudafy.GThread thread, float[] x, float[] y, float[] z, float[] w) {
            int tid = thread.blockIdx.x;
            w[tid] = ((((x[tid] * x[tid]) 
                        + (z[tid] * z[tid])) 
                        + (z[tid] * z[tid])) 
                        + -1);
        }
    }
}
```
If you would like to mess with the CodeDOM I would recommend using ```GenerateCSharpCode(CodeCompileUnit targetUnit, string fileName)``` to verfiy the CodeDOM is producing the code you want.

Some CodeDOM links for reading:
http://www.codeproject.com/Articles/20597/CodeDom-Assistant
http://www.codeproject.com/Articles/26312/Dynamic-Code-Integration-with-CodeDom
