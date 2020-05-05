#include <vector>
#include <fstream>

#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>

#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/util/Util.h>

#include "tiny_obj_loader.h"

int main(int argc, char* argv[])
{
    std::string root = argv[1];
    std::string subject = argv[2];

    using namespace openvdb;

    std::string filename = root + "/" + subject + "_OBJ_FBX/" + subject + "_100k.obj";
    std::cout << "loading obj from " << filename << std::endl;
    std::vector<Vec3f> points;
    std::vector<Vec3I> tris;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

    if (!err.empty()) {
      printf("err: %s\n", err.c_str());
    }

    if (!ret) {
      printf("failed to load : %s\n", filename.c_str());
      return -1;
    }

    if (shapes.size() == 0) {
      printf("err: # of shapes are zero.\n");
      return -1;
    }

    printf("# of vertices  = %d\n", (int)(attrib.vertices.size()) / 3);
    printf("# of normals   = %d\n", (int)(attrib.normals.size()) / 3);
    printf("# of texcoords = %d\n", (int)(attrib.texcoords.size()) / 2);
    printf("# of materials = %d\n", (int)materials.size());
    printf("# of shapes    = %d\n", (int)shapes.size());

    // Only use first shape.
    float s = 4.0;
    {
        for (size_t f = 0; f < shapes[0].mesh.indices.size()/3; f++) {
            int fd1 = shapes[0].mesh.indices[f*3+0].vertex_index;
            int fd2 = shapes[0].mesh.indices[f*3+1].vertex_index;
            int fd3 = shapes[0].mesh.indices[f*3+2].vertex_index;
            tris.push_back(Vec3I(fd1, fd2, fd3));
        }

        for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
            points.push_back(Vec3f(s*attrib.vertices[3*v+0], s*attrib.vertices[3*v+1], s*attrib.vertices[3*v+2]));
        }
    }

    math::Transform::Ptr xform = math::Transform::createLinearTransform();

    float sigma = 20.0;
    FloatGrid::Ptr grid = tools::meshToSignedDistanceField<FloatGrid>(*xform, points, tris, std::vector< Vec4I>(), sigma*s, sigma*s);
    
    std::vector<openvdb::Vec3s> verts;
    std::vector<openvdb::Vec4I> quads;
    tris.clear();

    openvdb::tools::VolumeToMesh mesher;
    mesher.operator()<openvdb::FloatGrid>( *grid );
    float isovalue = 0;
    float adaptivity = 0;
    openvdb::tools::volumeToMesh(*grid, verts, tris, quads, isovalue, adaptivity);

    std::string obj_out = root + "/" + subject + "_sdf.obj";
    std::string data_out = root + "/" + subject + "_sdf.data";
    std::ofstream fout(obj_out);
    for (openvdb::Vec3s vert : verts)
    {
        fout << "v " << (float)vert[0]/s << " " << (float)vert[1]/s << " " << (float)vert[2]/s << std::endl;
    }
    for (openvdb::Vec4I quad : quads)
    {
        fout << "f " << quad[0]+1 << " " << quad[2]+1 << " " << quad[1]+1 << std::endl;
        fout << "f " << quad[0]+1 << " " << quad[3]+1 << " " << quad[2]+1 << std::endl;
    }
    for (openvdb::Vec3I tri : tris)
    {
        fout << "f " << tri[0]+1 << " " << tri[2]+1 << " " << tri[1]+1 << std::endl;
    }
    fout.close();

    FILE *out;
    out = fopen(data_out.c_str(), "w+b");
    fseek(out, sizeof(int), 0);
    float data[4];
    
    int cnt = 0;
    for (openvdb::FloatGrid::ValueOnIter iter = grid->beginValueOn(); iter; ++iter) {
        float dist = iter.getValue() / sigma / s;
        auto coord = iter.getCoord();
        data[0] = (float)coord[0] / s;
        data[1] = (float)coord[1] / s;
        data[2] = (float)coord[2] / s;
        data[3] = dist;
        fwrite(data, sizeof(float), 4, out);
        cnt++;
    }
    fclose(out);

    out = fopen(data_out.c_str(), "r+b");
    fwrite(&cnt, sizeof(int), 1, out);
    fclose(out);

    std::cout << data[0] << " " << data[1] << " " << data[2] << " " << data[3] << std::endl;
    std::cout << "#points: " << cnt << std::endl;

    return 0;
}