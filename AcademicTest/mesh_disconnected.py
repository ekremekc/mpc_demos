import gmsh
import os
import sys
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

geom_dir = "/GeomDir/"
mesh_dir = "/MeshDir"
mesh_name = "/Disconnected"

def fltk_options():

    # Type of entity label (0: description,
    #                       1: elementary entity tag,
    #                       2: physical group tag)
    gmsh.option.setNumber("Geometry.LabelType", 2)

    gmsh.option.setNumber("Geometry.PointNumbers", 0)
    gmsh.option.setNumber("Geometry.LineNumbers", 0)
    gmsh.option.setNumber("Geometry.SurfaceNumbers", 2)
    gmsh.option.setNumber("Geometry.VolumeNumbers", 2)

    # Mesh coloring(0: by element type, 1: by elementary entity,
    #                                   2: by physical group,
    #                                   3: by mesh partition)
    gmsh.option.setNumber("Mesh.ColorCarousel", 0)

    gmsh.option.setNumber("Mesh.Lines", 0)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0) # CHANGE THIS FLAG TO 0 TO SEE LABELS

    gmsh.option.setNumber("Mesh.VolumeEdges", 2)
    gmsh.option.setNumber("Mesh.VolumeFaces", 2)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

gmsh.model.add("HR")
gmsh.option.setString('Geometry.OCCTargetUnit', 'M')

path = os.path.dirname(os.path.abspath(__file__))

gmsh.model.occ.importShapes(path+ geom_dir+'Inner.step')
gmsh.model.occ.synchronize()
gmsh.model.occ.importShapes(path+ geom_dir+'Outer.step')
gmsh.model.occ.synchronize()


# gmsh.model.occ.removeAllDuplicates()
# gmsh.model.occ.synchronize()

vol_tags=gmsh.model.getEntities(dim=3)
print(vol_tags)
"""
total_tag = np.arange(1,5)
total_tag.tolist()
hole_tags = np.arange(4,6)
hole_tags.tolist()
other_tags = list(set(total_tag) - set(hole_tags))
print("HOLE TAGS:", hole_tags)
print("OTHER TAGS:", other_tags)
"""

gmsh.model.addPhysicalGroup(3, [1], tag=1) # Inner
gmsh.model.addPhysicalGroup(3, [2], tag=2) # Outer

import numpy as np


surfaces = gmsh.model.occ.getEntities(dim=2)
print(surfaces)
inlet, inlet_mark = [], 1
outlet, outlet_mark = [], 2

for surface in surfaces:
    """
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    print(com)
    if np.isclose(com[2], [0]): 
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], inlet_mark)
        gmsh.model.setPhysicalName(surface[0], inlet_mark, "inlet")

    elif np.isclose(com[2], [1.0]): 
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], outlet_mark)
        gmsh.model.setPhysicalName(surface[0], outlet_mark, "outlet")
    
    else:
    """
    gmsh.model.addPhysicalGroup(surface[0], [surface[1]])
        
lc = 0.010 # 0.01
"""
gmsh.model.mesh.field.add("Constant", 1)
gmsh.model.mesh.field.setNumbers(1, "VolumesList", hole_tags)
gmsh.model.mesh.field.setNumber(1, "VIn", lc / 5)
gmsh.model.mesh.field.setNumber(1, "VOut", lc)

gmsh.model.mesh.field.setAsBackgroundMesh(1)
"""
gmsh.model.occ.synchronize()


gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)#10
gmsh.option.setNumber("Mesh.RandomFactor", 1e-11)
gmsh.option.setNumber("Mesh.RandomFactor3D", 1e-13)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1)
gmsh.model.mesh.generate(3)


gmsh.write("{}.msh".format(dir_path +mesh_dir+mesh_name))

if '-nopopup' not in sys.argv:
    fltk_options()
    gmsh.fltk.run()

gmsh.finalize()


from helmholtz_x.dolfinx_utils import  write_xdmf_mesh

write_xdmf_mesh(dir_path +mesh_dir+mesh_name,dimension=3)


