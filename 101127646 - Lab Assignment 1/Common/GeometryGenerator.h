//***************************************************************************************
// GeometryGenerator.h by Frank Luna (C) 2011 All Rights Reserved.
//   
// Defines a static class for procedurally generating the geometry of 
// common mathematical objects.
//
// All triangles are generated "outward" facing.  If you want "inward" 
// facing triangles (for example, if you want to place the camera inside
// a sphere to simulate a sky), you will need to:
//   1. Change the Direct3D cull mode or manually reverse the winding order.
//   2. Invert the normal.
//   3. Update the texture coordinates and tangent vectors.
//***************************************************************************************

#pragma once

#include <cstdint>
#include <DirectXMath.h>
#include <vector>

class GeometryGenerator
{
public:

    using uint16 = std::uint16_t;
    using uint32 = std::uint32_t;

	struct Vertex
	{
		Vertex(){}
        Vertex(
            const DirectX::XMFLOAT3& p, 
            const DirectX::XMFLOAT3& n, 
            const DirectX::XMFLOAT3& t, 
            const DirectX::XMFLOAT2& uv) :
            Position(p), 
            Normal(n), 
            TangentU(t), 
            TexC(uv){}
		Vertex(
			float px, float py, float pz, 
			float nx, float ny, float nz,
			float tx, float ty, float tz,
			float u, float v) : 
            Position(px,py,pz), 
            Normal(nx,ny,nz),
			TangentU(tx, ty, tz), 
            TexC(u,v){}

        DirectX::XMFLOAT3 Position;
        DirectX::XMFLOAT3 Normal;
        DirectX::XMFLOAT3 TangentU;
        DirectX::XMFLOAT2 TexC;
	};

	struct MeshData
	{
		std::vector<Vertex> Vertices;
        std::vector<uint32> Indices32;

        std::vector<uint16>& GetIndices16()
        {
			if(mIndices16.empty())
			{
				mIndices16.resize(Indices32.size());
				for(size_t i = 0; i < Indices32.size(); ++i)
					mIndices16[i] = static_cast<uint16>(Indices32[i]);
			}

			return mIndices16;
        }

	private:
		std::vector<uint16> mIndices16;
	};

    MeshData CreateBox(float width, float height, float depth, uint32 numSubdivisions);


    MeshData CreateSphere(float radius, uint32 sliceCount, uint32 stackCount);

	// New
	MeshData CreateCone(float bottomRadius, float height, uint32 sliceCount, uint32 stackCount);

	MeshData GeometryGenerator::CreateTorus(float radius, float crossRadius, uint32 sliceCount, uint32 crossCount);

	MeshData CreatePyramid(float width, float height, float depth, uint32 numSubdivisions);

	MeshData CreateDiamond(float width, float height, float depth, uint32 numSubdivisions);

	MeshData GeometryGenerator::CreateTriangularPrism(float bottomRadius, float topRadius, float height, uint32 stackCount);

	MeshData CreateWedge(float width, float height, float depth, uint32 numSubdivisions);

	//End of New

    MeshData CreateGeosphere(float radius, uint32 numSubdivisions);

    MeshData CreateCylinder(float bottomRadius, float topRadius, float height, uint32 sliceCount, uint32 stackCount);


    MeshData CreateGrid(float width, float depth, uint32 m, uint32 n);

    MeshData CreateQuad(float x, float y, float w, float h, float depth);
	void Subdivide(MeshData& meshData);
private:
	
    Vertex MidPoint(const Vertex& v0, const Vertex& v1);
    void BuildCylinderTopCap(float bottomRadius, float topRadius, float height, uint32 sliceCount, uint32 stackCount, MeshData& meshData);
    void BuildCylinderBottomCap(float bottomRadius, float topRadius, float height, uint32 sliceCount, uint32 stackCount, MeshData& meshData);
	void BuildConeTopCap(float height, uint32 sliceCount, MeshData& meshData);

};
