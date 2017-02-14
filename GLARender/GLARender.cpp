// GLARender.cpp : Defines the entry point for the console application.
//

#include "cl.hpp"
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <windows.h>
#include <ctime>

#define MAX_NZ 512 

typedef struct ProgramExecutionContext
{
	cl::Program program;
	cl::Device  device;
	cl::Context context;
} ProgramExecutionContext;

void  get_program(ProgramExecutionContext &progCTX)
{ 

	const char* source_a = R"CL(
 
  #pragma OPENCL EXTENSION cl_amd_printf : disable
   #define WB 512

 float3 rotate_yz( float3 v)
{
   float3 rr;
   float  cos_theta = cos(0.7f) ;
   float  sin_theta = sin(0.7f) ;

    rr.z = v.z*cos_theta - v.y*sin_theta ;
    rr.y = v.z*sin_theta + v.y*cos_theta ;
    rr.x = v.x;
    
    return rr; 
}

float3 rotate_xy( float3 v)
{
   float3 rr;
   float  cos_theta = cos(0.3f) ;
   float  sin_theta = sin(0.3f) ;

    rr.x = v.x*cos_theta - v.y*sin_theta ;
    rr.y = v.x*sin_theta + v.y*cos_theta ;
    rr.z = v.z;
    
    return rr; 
}
 float3 rotate_w( float3 v)
{
  return rotate_yz(rotate_xy(v));
}

    void  get_local_velocity(   local float3 *inputROZ , float3 ray  , local float  *outputVelTangencial      ) 
          {
            
            int j   = (get_local_size( 0 ) * get_local_id(1)) + get_local_id(0) ; // offset local 
            float3 xyz = inputROZ[ j ];   
            float r  = length ( xyz  );         
            float3 axis_z =  { 0  , 0.0f,  1.0f  } ;
            float3 axis_vel =  normalize(cross(axis_z , xyz));
            float vm =   3.0f*atan( ( r / 10.0f ) );  // modulo da velocidade
            outputVelTangencial[j]  = dot(axis_vel , ray) * vm; // velocidade visual       
 
          }


 

 void  get_local_luminosity_v(   local float3 *inputROZ ,   local float  *inputVelTangencial ,   local float  *outputLum_v  ,   float channel_v   ) 
  {
               int j   = (get_local_size( 0 ) * get_local_id(1)) + get_local_id(0) ; // offset local
               float3 xyz = inputROZ[ j ];   
               float r  = length ( xyz  );     

               
               float3 axis_z =  { 0, 0.0f,  1.0f  } ;
               float3 axis_y =  { 0.0f , 1.0f, 0.0f } ;

               float  local_v = inputVelTangencial[j] ;

               float delta_v = fabs(local_v - channel_v)/0.2f;
			   float conv_v = 0.0f;
			   if (delta_v < 3.0)
			   {
				  conv_v = exp(-1.0f* delta_v );

			   }
			   float dd =   1.0f/(r*r  + 10.0f);   
              // if (r > 10 ) dd = 0.0 ;
			   float plane_d = fabs(  xyz.z  ); 
               // if (plane_d >= 2.0) dd =0.0;
			   dd = dd /(plane_d*plane_d + 0.2); 
			   outputLum_v[ j ] = conv_v *    dd;  
  }
 
 

  void get_localCoordinades_inn(global const float2 *inputXY,    const int nz ,    local float3 *local_xyz)
   {
         int tixy = get_global_id(0); //indice do XY 
         int tiz = get_global_id(1); // indice do NZ

        

         int ixy = get_local_id(0);         
           

         int j   = (get_local_size( 0 ) * get_local_id(1)) + get_local_id(0) ; // offset local 
         {            
           float local_z =   1.0f*tiz ;
           local_z =  local_z/nz  ;
           local_z = 16.0f * local_z -16.0f  ;
           float3 xyz = {inputXY[tixy].x  ,inputXY[tixy].y,  local_z } ;  
           
           local_xyz[ j ] = rotate_w(xyz);

         //  if(j==0) printf((__constant char *)"%4f -> %4f %4f  %4f \n",local_z, local_xyz[j].x , local_xyz[j].y , local_xyz[j].z );
         }

         
    }

 kernel void computeLuminosity(global const float2 *inputXY,    const int nz , const float2 channel_range,    global float *outputLum , global float *outputLumChannel)
 {
      

      int tixy = get_global_id(0); //indice do XY 
      int tiz = get_global_id(1); // indice do NZ 
       
        __local float3 local_xyz[WB]; 
        __local float  local_vvv[WB]; 
        __local float  local_lum_v[WB];

      //printf((__constant char *)"%4f %4f == %4f %4f  %4f \n", inputXY[tixy].x , inputXY[tixy].y , x0,y0,z0);
        
        //processa as posicoes 
        get_localCoordinades_inn( inputXY ,  nz ,local_xyz  );

        //processa a velocidade de cada posicao xyz             

        //velocidades locais
         
        float3 ray =  rotate_w( ( float3){ 0.0f  , 0.0f,  1.0f  });
        get_local_velocity(  local_xyz ,ray , local_vvv   );

        //luminosidade local  
         
        float channel_v = channel_range.x;
        get_local_luminosity_v( local_xyz, local_vvv ,   local_lum_v ,   channel_v   ) ;
        


        barrier(CLK_LOCAL_MEM_FENCE); 
        if (get_local_id(1) ==0 ) // Somatoria em Z ocorre em Z == 0
        {
          float acc = 0.0;
          for(int  jz = 0  ; jz <  get_local_size(1); ++jz )
          {
               int j   = (get_local_size( 0 ) * jz ) + get_local_id(0) ; // offset local                
               acc = acc +  local_lum_v[j];
          }
          outputLum[tixy] = acc ;
          int channel_i = (channel_v + 3.6)/0.02f;
          //channel_i = min(channel_i,100);
          int mapOffset =  get_global_size(0) * channel_i ;
         // outputLumChannel[  mapOffset + tixy ]  = acc ;
          outputLumChannel[  tixy ]  = acc ;
        } 


 }



)CL";

 



	cl::Program vectorWrapper(progCTX.context,	cl::STRING_CLASS(source_a), false);

	

	{
		
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		int i = 0;
		for (cl::Platform const& platform : platforms) {
			std::string platform_name;
			std::string platform_version;
			std::string platform_extensions;

			platform.getInfo(CL_PLATFORM_NAME, &platform_name);
			platform.getInfo(CL_PLATFORM_VERSION, &platform_version);
			platform.getInfo(CL_PLATFORM_EXTENSIONS, &platform_extensions);

			std::cout << "#" << i << std::endl;
			std::cout << "name: " << platform_name << std::endl;
			std::cout << "version: " << platform_version << std::endl;
			std::cout << "extensions: " << platform_extensions << std::endl;
			std::cout << "============" << std::endl;
			++i;

		 

		}
		size_t size_cp_max;
		cl::Device::getDefault().getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &size_cp_max);
		std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << size_cp_max << std::endl;
	 
		  std::string  spir_versions;
		 cl::Device::getDefault().getInfo(CL_DEVICE_SPIR_VERSIONS, &spir_versions);
		  
		 {
			 std::cout << "CL_DEVICE_SPIR_VERSIONS: " << spir_versions << std::endl;
		 }
	}

	{
		cl_int c_err =  vectorWrapper.compile("-g -x spir -spir-std=1.2"  );

	
		{
			cl_int err_bin = 0;
			auto binary = vectorWrapper.getInfo<CL_PROGRAM_BINARIES>(&err_bin);			
			auto sizes = vectorWrapper.getInfo<CL_PROGRAM_BINARY_SIZES>();

			for (auto b : sizes)
			{
				
				std::cout << "Binary Size : " <<  b  << std::endl;
				continue;
				for(int j =0 ;j< b;++j)
				{
					printf("%c", binary.front()[j]);
					if (j % 40 == 0)
					{
						printf("\n");
					}
				}
			}
		}

	 
	 
		if (c_err != CL_SUCCESS) {
			std::cout << "Build Status: " << vectorWrapper.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl::Device::getDefault()) << std::endl;
			std::cout << "Build Options:\t" << vectorWrapper.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl::Device::getDefault()) << std::endl;
			std::cout << "Build Log:\t " << vectorWrapper.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()) << std::endl;
			exit(1);
		}


		//vectorWrapper.build("-cl-std=CL2.0");
		std::string bl = vectorWrapper.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault());
		std::cout << bl << std::endl;
	}
	 
		std::vector<cl::Program> programs;
		 
		programs.push_back(vectorWrapper);
		cl::Program vectorProgram = cl::linkProgram(programs);

		

 
		progCTX.program = vectorProgram;
	 
	 
		return;
}

cl::make_kernel< cl::Buffer&,   int, cl_float2, cl::Buffer&, cl::Buffer&>   getLuminosityFunction(cl::Program vectorProgram)
{
	//kernel void computeLuminosity(global const float2 *inputXY,  const float3 ray_x ,  const float3 ray_y ,   const float3 ray_z ,  const int nz , const float2 channel_range,    global float3 *outputLum)

	static  auto coordinadesKernel =
		cl::make_kernel<
		cl::Buffer&,
	 
		int,
		cl_float2,
		cl::Buffer&, 
		cl::Buffer&
		>(vectorProgram, "computeLuminosity");

	return coordinadesKernel;

}

 

cl::make_kernel< cl::Buffer&,   int, cl::Buffer&>   getfillDensity(cl::Program vectorProgram)
{

	static auto luminosityKernel =
		 cl::make_kernel<
		 cl::Buffer&,		
		 int,
		 cl::Buffer&
		 >(vectorProgram, "fillDensity");


	 return   luminosityKernel;

}
 


cl_float3 rotateRay(cl_float3 ray, float __WA, float __WB)
{
	 
	{
		float wa = __WB;;
		float wb = __WA;

		{
			float dx = ray.x * cos(wa) - ray.y * sin(wa);
			float dy = ray.x * sin(wa) + ray.y * cos(wa);
			ray.x = dx;
			ray.y = dy;
		}
		{
			float dy = ray.y * cos(wb) - ray.z * sin(wb);
			float dz = ray.y * sin(wb) + ray.z * cos(wb);
			ray.y = dy;
			ray.z = dz;
		}
		 
		return {   ray.x,   ray.y,  ray.z  };

	}
}
 
void resolveGalaticalCoordinades__(cl::CommandQueue&  queue , std::vector<cl::Buffer> &buffers , cl::Program vectorProgram ,float x1, float x2, float y1, float y2, float dxy ,float __WA , float __WB, cl_float2 channel_range)
{

	int ny = fabs(y2 - y1) / dxy;
	int nx = fabs(x2 - x1) / dxy;

	int tilesize = 32;

	int batchSize = nx*ny; // num shaders to compute por batch
	int nz = MAX_NZ;

	while (batchSize % 16 != 0) batchSize++;

	 static auto functionKernel = getLuminosityFunction(vectorProgram);
	 

	//std::vector< cl_float2 > inputXY(batchSize, cl_float2()); // entrada eh a tela
	//std::vector< cl_float3 > coordinadesROZ(batchSize * nz,{ -1.0f, -1.0f, -1.0f}); // saida sao as tres coordenadas galaticas
	//std::vector< cl_float  > densityXYZ(batchSize * nz, 0.0f);
	//std::vector< cl_float  > densityL(batchSize , 0.0f);

	 


	cl::Buffer& inputBBuffer = buffers[0]; 
	cl::Buffer& densityLBuffer = buffers[1];
	cl::Buffer& densityLVBuffer = buffers[2];

	cl_float3 ray_x = { 1.0f,0,0 };
	cl_float3 ray_y = { 0,1.0f,0 };
	cl_float3 ray_z = { 0,0,1.0f };
	//ray_z = rotateRay(ray_z,__WA, __WB);
	//ray_y = rotateRay(ray_y, __WA, __WB);
	//ray_x = rotateRay(ray_x, __WA, __WB);
	
 

	size_t size_w_max;
	cl::Device::getDefault().getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size_w_max); 
	std::vector<size_t> deviceSizes;
	cl::Device::getDefault().getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &deviceSizes);
	//std::cout << "max Work items Size  " << deviceSizes[0]<<", "<< deviceSizes[1] << std::endl;

	//std::cout << "max Work Size  " << size_w_max << std::endl;

	int LC = deviceSizes[0];
	while ((LC*LC) > size_w_max) LC = LC / 2;

	auto v = functionKernel;

 

	//cl::NDRange globalRange = cl::NDRange(batchSize, nz);
	cl::NDRange globalRange = cl::NDRange(batchSize, nz);
	cl::NDRange localRange = cl::NDRange(16, 16);
 
	int workItens = (batchSize * nz) / 64;

	functionKernel(
	cl::EnqueueArgs( queue,
	cl::NDRange (globalRange),
	cl::NDRange(localRange)),
		inputBBuffer,	   
		nz,
		channel_range,
		densityLBuffer, densityLVBuffer);

 
	//passa para o proximo estagio
	//cl::copy(coordinadesBuffer, begin(coordinadesROZ), end(coordinadesROZ));
	 
 //

	// 
	//	auto functionKernel_2 = getDepthDensity(vectorProgram);
	//	functionKernel_2(
	//		cl::EnqueueArgs(globalRange, localRange),
	//		coordinadesBuffer,
	//		channel_range, 
	//		ray_z,
	//		nz,
	//		densityXYZBuffer);
 //
	// 
	////show results


	//	auto functionKernel_3 = getfillDensity( vectorProgram);
	//	functionKernel_3(
	//		cl::EnqueueArgs(batchSize, size_w_max),
	//		densityXYZBuffer,
	//		nz,
	//		densityLBuffer);




	// 

	

}

void resolveGalaticalCoordinades(ProgramExecutionContext &progCTX, float x1, float x2, float y1, float y2, float dxy)
{


	auto  vectorProgram = progCTX.program;


	int ny = fabs(y2 - y1) / dxy;
	int nx = fabs(x2 - x1) / dxy;
	
	float  chanel_range_v = 3.6;
	float  delta_chanel_range_v = 0.02f;

	int num_channels = (chanel_range_v * 2.0) / delta_chanel_range_v;


	 

	int tilesize = 32;

	int batchSize = (nx )*(ny ); // num shaders to compute por batch
	int nz = MAX_NZ;

	while (batchSize % 16 != 0) batchSize++;




	//auto functionKernel = getRotateFunction(vectorProgram);
	int nchannels = num_channels ;
	
	

	std::vector< cl_float2 > inputXY(batchSize, cl_float2()); // entrada eh a tela
	//std::vector< cl_float3 > coordinadesROZ(batchSize * nz, { -1.0f, -1.0f, -1.0f }); // saida sao as tres coordenadas galaticas
	//std::vector< cl_float  > densityXYZ(batchSize * nz, 0.0f);
	std::vector< cl_float  > densityL(batchSize, 0.0f);
	std::vector< cl_float  > densityLV_a(batchSize , 0.0f);
	std::vector< cl_float  > densityLV_b(batchSize , 0.0f);


	// monta o tile do calculo
	{
		srand(time(NULL));

		/* generate secret number between 1 and 10: */
	 
		int xmm = (x2 + x1) / 2.0f;
		int ymm = (y2 + y1) / 2.0f;
		int i = 0;
		for (int yt = 0; yt <   ny ; yt += 1)
			for (int xt = 0; xt <   nx ; xt += 1)
			{				 
				inputXY[i].x = xmm + (xt - nx/2.0f) * dxy ;
				inputXY[i].y = ymm + (yt - ny/2.0f) * dxy ;
				 // printf("%5.2f %5.2f \n", inputXY[i].x, inputXY[i].y);
				++i;
			}
	 
	}
	
 

 

	cl::CommandQueue queue_1(progCTX.context, progCTX.device);
	cl::CommandQueue queue_2(progCTX.context, progCTX.device);

	
	cl::Buffer inputBBuffer(progCTX.context,begin(inputXY), end(inputXY), true);
	//cl::Buffer coordinadesBuffer(begin(coordinadesROZ), end(coordinadesROZ), false);
	//cl::Buffer densityXYZBuffer(begin(densityXYZ), end(densityXYZ), false);
	cl::Buffer densityLBuffer(progCTX.context, begin(densityL), end(densityL), false);
	cl::Buffer densityLVBuffer_a(progCTX.context, begin(densityLV_a), end(densityLV_a), false);
	cl::Buffer densityLVBuffer_b(progCTX.context, begin(densityLV_b), end(densityLV_b), false);
	

	const WORD colors[] =
	{
		0x04, 0x06, 0x02, 0x03, 0x09, 0x01,
		0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6
	};

	HANDLE hstdin = GetStdHandle(STD_INPUT_HANDLE);
	HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
	
	WORD   index = 0;

	std::vector<cl::Buffer> buffers_a = { inputBBuffer , densityLBuffer , densityLVBuffer_a  };
	std::vector<cl::Buffer> buffers_b = { inputBBuffer , densityLBuffer , densityLVBuffer_b };

	int jloop = 0;
	
	printf("start \n");
	for (float channel = -chanel_range_v; channel < chanel_range_v; channel += delta_chanel_range_v)
	{
		cl_float2 channel_range = { channel ,channel + 1.0f };
		//resolveGalaticalCoordinades__(buffers, vectorProgram, x1, x2, y1, y2, dxy, channel, 0.0f, channel_range);
		printf(".");
	}

	 

	printf("done\n");
	printf("start copy\n");
	cl::copy(queue_1,densityLVBuffer_a, begin(densityLV_a), end(densityLV_a));
	printf("end copy\n");


	 

	 
	int q_idx = 1; // 1 ou 2 
	

	for (float channel = -chanel_range_v; channel < chanel_range_v; channel += delta_chanel_range_v)
	{
		cl_float2 channel_range = { channel ,channel+1.0f };	
		//cl::copy(densityLBuffer, begin(densityL), end(densityL));
		
		std::vector< float> densityL_VI;
		if (q_idx == 1)
		{
			resolveGalaticalCoordinades__(queue_1, buffers_a, vectorProgram, x1, x2, y1, y2, dxy, channel, 0.0f, channel_range);
			cl::copy(queue_2, densityLVBuffer_b, begin(densityLV_b), end(densityLV_b));
			queue_2.finish();
			densityL_VI = std::vector< float>(densityLV_b.begin(), densityLV_b.begin() + batchSize);
			q_idx = 2;
		}
		else if (q_idx == 2)
		{
			resolveGalaticalCoordinades__(queue_2, buffers_b, vectorProgram, x1, x2, y1, y2, dxy, channel, 0.0f, channel_range);
			cl::copy(queue_1, densityLVBuffer_a, begin(densityLV_a), end(densityLV_a));
			queue_1.finish();
			densityL_VI = std::vector< float>(densityLV_a.begin(), densityLV_a.begin() + batchSize);
			q_idx = 1;
		}
		
	 

		//int offset_map = (channel + chanel_range_v) / delta_chanel_range_v;
		//offset_map = offset_map * batchSize;
		
		 
		
		jloop++;

		
		//if(jloop%10 == 0 )
		{
			auto Lmax_it = std::max_element(begin(densityL_VI), end(densityL_VI));
			auto Lmax = *Lmax_it;

			if (Lmax < 1e-10)
			{
				Lmax = 1e-10;
				continue;
			}

			 printf("channel %4f  L Max %f \n", channel , Lmax);
		 
			 //continue;
			int  i = 0;

			CONSOLE_SCREEN_BUFFER_INFO csbi;
			GetConsoleScreenBufferInfo(hstdout, &csbi);
			DWORD dwConSize = csbi.dwSize.X * csbi.dwSize.Y;
			COORD coordScreen = { 0, 0 };
			DWORD cCharsWritten;

			CHAR_INFO *buffer = static_cast<CHAR_INFO*>(malloc(sizeof(CHAR_INFO)*csbi.dwSize.Y*csbi.dwSize.X));

			 
				HANDLE hOutput = static_cast<HANDLE>(GetStdHandle(STD_OUTPUT_HANDLE));
				COORD dwBufferSize = { csbi.dwSize.X,csbi.dwSize.Y };
				COORD dwBufferCoord = { 0, 0 };
				SMALL_RECT rcRegion = { 0, 0, csbi.dwSize.X - 1, csbi.dwSize.Y - 1 };
				
				CHAR_INFO zer0_cc;
				zer0_cc.Attributes = colors[0];
				zer0_cc.Char.AsciiChar = ' ';
				memset(buffer , 0, csbi.dwSize.Y*csbi.dwSize.X);
				ReadConsoleOutput(hOutput,  buffer, dwBufferSize,	dwBufferCoord, &rcRegion);

			    for(int j = 0; j <csbi.dwSize.X * csbi.dwSize.Y; j++)
			    {
					buffer[j].Attributes = colors[0];  buffer[j].Char.AsciiChar = ' ';
			    }

		 
			// SetConsoleTextAttribute(hstdout, 0x0F);  
			// FillConsoleOutputCharacter(hstdout, ' ', dwConSize, coordScreen, &cCharsWritten);
			 //printf("--------------------------------------------------------------------------------------------\n");

			 for (int yt = -ny / 2; yt <  ny / 2; yt += 1)
			 {
				 for (int xt = -nx / 2; xt <  nx / 2; xt += 1)
				{
					//if (i >= batchSize) break;
					float LL = densityL_VI[i];
					++i;

					

					if (LL >= (Lmax / 16.0f))
					{
						COORD pxy_coord;
						pxy_coord.X = (csbi.dwSize.X*(xt + nx / 2))/nx ;
						pxy_coord.Y = (csbi.dwSize.Y*(yt + ny / 2))/ny ;
						
						pxy_coord.X = min(pxy_coord.X, csbi.dwSize.X - 1);
						pxy_coord.Y = min(pxy_coord.Y, csbi.dwSize.Y - 1);

						int k = (pxy_coord.Y * csbi.dwSize.X) + pxy_coord.X;

						//SetConsoleCursorPosition(hstdout, pxy_coord);
			/*			if (LL >= (Lmax / 2.0f)) { SetConsoleTextAttribute(hstdout, colors[0]);  printf("X"); continue; }
						if (LL >= (Lmax / 4.0f)) { SetConsoleTextAttribute(hstdout, colors[1]); printf("*"); continue; }
						if (LL >= (Lmax / 6.0f)) { SetConsoleTextAttribute(hstdout, colors[2]); printf("*"); continue; }
						if (LL >= (Lmax / 8.0f)) { SetConsoleTextAttribute(hstdout, colors[3]); printf(":"); continue; }
						if (LL >= (Lmax / 12.0f)) { SetConsoleTextAttribute(hstdout, colors[4]); printf(":"); continue; }
						if (LL >= (Lmax / 16.0f)) { SetConsoleTextAttribute(hstdout, colors[5]); printf("."); continue; }*/


						if (LL >= (Lmax / 2.0f)) { buffer[k].Attributes = colors[0];  buffer[k].Char.AsciiChar ='X'; continue; }
						if (LL >= (Lmax / 4.0f)) { buffer[k].Attributes = colors[1]; buffer[k].Char.AsciiChar = '*'; continue; }
						if (LL >= (Lmax / 6.0f)) { buffer[k].Attributes = colors[2]; buffer[k].Char.AsciiChar = '*'; continue; }
						if (LL >= (Lmax / 8.0f)) { buffer[k].Attributes = colors[3]; buffer[k].Char.AsciiChar = ':'; continue; }
						if (LL >= (Lmax / 12.0f)) { buffer[k].Attributes = colors[4]; buffer[k].Char.AsciiChar = ':'; continue; }
						if (LL >= (Lmax / 16.0f)) { buffer[k].Attributes = colors[5]; buffer[k].Char.AsciiChar = '.'; continue; }
						buffer[k].Char.AsciiChar = ' ';


						

					}
					//printf(" ");
					

				}
				 //printf("\n");
				
			}

			 WriteConsoleOutput(hOutput, buffer, dwBufferSize,	 dwBufferCoord, &rcRegion);

	 
			Lmax += 1.0f;
			//printf("--------------------------------------------------------------------------------------------");
			//printf("\n");
			
		}
	}
 
	return;
}



int main()
{
	//executeCLProgram(32);
 
	std::vector<cl::Platform> platforms;

	try{
		cl_int  err = cl::Platform::get(&platforms);
	 
		printf("number of platforms: %d\n", platforms.size());
		if (platforms.size() == 0) {
			printf("Platform size 0\n");
		}
	}
	catch(...)
	{
		printf("Platform error  0\n");
	}
 


	cl_context_properties properties[] =
	{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>((platforms[0])()), 0 };
	cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

	 
 

 
 
	cl::Device default_device = context.getInfo<CL_CONTEXT_DEVICES>()[0];


	 
	ProgramExecutionContext progCTX;

	progCTX.context = context;
	progCTX.device = default_device;

   get_program(progCTX);
   resolveGalaticalCoordinades(progCTX, -32, 32, -20, 20, 0.2f);

	 
    return 0;
}

