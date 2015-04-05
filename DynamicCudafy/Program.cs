using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.CodeDom;
using System.CodeDom.Compiler;
using System.Reflection;
using Microsoft.CSharp;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
namespace DynamicCudafy
{
    class Program
    {
        delegate void PrintDelegate();
        delegate void ScalarDelegate(GThread t,float[] x,float[] y,float[] z, float[] w);
        static void Main(string[] args)
        {
            //Code generator
            CodeCompileUnit targetUnit = new CodeCompileUnit();

            //Namespace and imports
            CodeNamespace dynGpu = new CodeNamespace("DynamicGPU");
            dynGpu.Imports.Add(new CodeNamespaceImport("System"));
            dynGpu.Imports.Add(new CodeNamespaceImport("Cudafy"));
            dynGpu.Imports.Add(new CodeNamespaceImport("Cudafy.Host"));
            dynGpu.Imports.Add(new CodeNamespaceImport("Cudafy.Translator"));
            targetUnit.Namespaces.Add(dynGpu);

            //Class declaration
            CodeTypeDeclaration dualMarchingCubes = new CodeTypeDeclaration("DualMarchingCubes");
            dualMarchingCubes.Attributes = MemberAttributes.Public;
            dynGpu.Types.Add(dualMarchingCubes);

            //ScalarField Function
            CodeMemberMethod ScalarField = new CodeMemberMethod()
            {
                Attributes = MemberAttributes.Public | MemberAttributes.Static,
                Name = "ScalarField",
                ReturnType = new CodeTypeReference(typeof(void)),
                CustomAttributes = new CodeAttributeDeclarationCollection() { new CodeAttributeDeclaration("Cudafy") }
            };

            ScalarField.Parameters.Add(new CodeParameterDeclarationExpression(typeof(GThread), "thread"));
            ScalarField.Parameters.Add(new CodeParameterDeclarationExpression(typeof(float[]), "x"));
            ScalarField.Parameters.Add(new CodeParameterDeclarationExpression(typeof(float[]), "y"));
            ScalarField.Parameters.Add(new CodeParameterDeclarationExpression(typeof(float[]), "z"));
            ScalarField.Parameters.Add(new CodeParameterDeclarationExpression(typeof(float[]), "w"));
            dualMarchingCubes.Members.Add(ScalarField);

            CodeVariableReferenceExpression thread = new CodeVariableReferenceExpression()
            {
                VariableName = "thread"
            };
            CodePropertyReferenceExpression _thread_blockIdx = new CodePropertyReferenceExpression()
            {
                PropertyName = "blockIdx",
                TargetObject = thread
            };
            CodePropertyReferenceExpression _thread_blockIdx_x = new CodePropertyReferenceExpression()
            {
                PropertyName = "x",
                TargetObject = _thread_blockIdx
            };

            CodeVariableDeclarationStatement tid = new CodeVariableDeclarationStatement(typeof(int), "tid")
            {
                InitExpression = _thread_blockIdx_x
            };
            ScalarField.Statements.Add(tid);

            CodeIndexerExpression xIdx = new CodeIndexerExpression();
            xIdx.Indices.Add(new CodeVariableReferenceExpression("tid"));
            CodeVariableReferenceExpression x = new CodeVariableReferenceExpression("x");
            xIdx.TargetObject = x;
            CodeBinaryOperatorExpression x2 = new CodeBinaryOperatorExpression(xIdx, CodeBinaryOperatorType.Multiply, xIdx);

            CodeIndexerExpression yIdx = new CodeIndexerExpression();
            yIdx.Indices.Add(new CodeVariableReferenceExpression("tid"));
            CodeVariableReferenceExpression y = new CodeVariableReferenceExpression("z");
            yIdx.TargetObject = y;
            CodeBinaryOperatorExpression y2 = new CodeBinaryOperatorExpression(yIdx, CodeBinaryOperatorType.Multiply, yIdx);

            CodeIndexerExpression zIdx = new CodeIndexerExpression();
            zIdx.Indices.Add(new CodeVariableReferenceExpression("tid"));
            CodeVariableReferenceExpression z = new CodeVariableReferenceExpression("z");
            zIdx.TargetObject = z;
            CodeBinaryOperatorExpression z2 = new CodeBinaryOperatorExpression(zIdx, CodeBinaryOperatorType.Multiply, zIdx);

            CodeBinaryOperatorExpression x2y2 = new CodeBinaryOperatorExpression(x2, CodeBinaryOperatorType.Add, y2);
            CodeBinaryOperatorExpression x2y2z2 = new CodeBinaryOperatorExpression(x2y2, CodeBinaryOperatorType.Add, z2);
            CodeBinaryOperatorExpression sum = new CodeBinaryOperatorExpression(x2y2z2, CodeBinaryOperatorType.Add, new CodePrimitiveExpression(-1));

            CodeIndexerExpression wIdx = new CodeIndexerExpression();
            wIdx.Indices.Add(new CodeVariableReferenceExpression("tid"));
            CodeVariableReferenceExpression w = new CodeVariableReferenceExpression("w");
            wIdx.TargetObject = w;

            CodeAssignStatement result = new CodeAssignStatement(wIdx, sum);
            ScalarField.Statements.Add(result);

            GenerateCSharpCode(targetUnit, "codes.cs");

            Assembly gpuAssy = CompileDOM(targetUnit);
            object dmc = gpuAssy.CreateInstance("DynamicGPU.DualMarchingCubes");
            Type dmcType = gpuAssy.GetType("DynamicGPU.DualMarchingCubes");
            MethodInfo scalarFunction = dmcType.GetMethod("ScalarField");



            GPGPU gpu = CudafyHost.GetDevice(eGPUType.OpenCL);
            CudafyModule km = CudafyTranslator.Cudafy(ePlatform.Auto, gpu.GetArchitecture(), dmcType);
            gpu.LoadModule(km);
            int N = 100000000;
            Random r = new Random();
            float[] _x = new float[N];
            float[] _y = new float[N];
            float[] _z = new float[N];

            float[] dev_x = gpu.CopyToDevice(_x);
            float[] dev_y = gpu.CopyToDevice(_y);
            float[] dev_z = gpu.CopyToDevice(_z);
            float[] dev_w = gpu.Allocate<float>(N);

            Console.WriteLine("Ready...");
            Console.ReadKey();
            gpu.Launch(N/2, 2).ScalarField(dev_x, dev_y, dev_z, dev_w);
            float[] _w = new float[N];
            gpu.CopyFromDevice(dev_w, _w);
            Console.WriteLine("Completed!");
            Console.ReadKey();
        }

        private static Assembly CompileSource(string sourceCode)
        {
            CodeDomProvider cpd = new CSharpCodeProvider();
            CompilerParameters cp = new CompilerParameters();
            cp.ReferencedAssemblies.Add("System.dll");
            cp.ReferencedAssemblies.Add("Cudafy.NET.dll");
            cp.GenerateExecutable = false;
            // Invoke compilation.
            CompilerResults cr = cpd.CompileAssemblyFromSource(cp, sourceCode);
            if (cr.Errors.Count > 0)
            {
                for (int i = 0; i < cr.Errors.Count; i++)
                {
                    Console.WriteLine(cr.Errors[i].ErrorText);
                }
                throw new ArgumentNullException();
            }
            return cr.CompiledAssembly;
        }

        private static Assembly CompileDOM(CodeCompileUnit cU)
        {
            CodeDomProvider cpd = new CSharpCodeProvider();
            CompilerParameters cp = new CompilerParameters();
            cp.ReferencedAssemblies.Add("System.dll");
            cp.ReferencedAssemblies.Add("Cudafy.NET.dll");
            cp.GenerateExecutable = false;
            CompilerResults cr = cpd.CompileAssemblyFromDom(cp, cU);
            if (cr.Errors.Count > 0)
            {
                for (int i = 0; i < cr.Errors.Count; i++)
                {
                    Console.WriteLine(cr.Errors[i].ErrorText);
                }
                throw new ArgumentNullException();
            }
            return cr.CompiledAssembly;
        }



        static public void GenerateCSharpCode(CodeCompileUnit targetUnit, string fileName)
        {
            CodeDomProvider provider = new CSharpCodeProvider();
            CodeGeneratorOptions options = new CodeGeneratorOptions();
            //options.BracingStyle = "C";
            using (var fileStream = new StreamWriter(File.Create(fileName)))
            {
                provider.GenerateCodeFromCompileUnit(targetUnit, fileStream, null);
            }
            
        }
    }
}
