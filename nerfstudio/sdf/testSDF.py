from pysdf import SDF

# Load some mesh (don't necessarily need trimesh)
import trimesh
o = trimesh.load('glass_tea.obj',force="mesh")
f = SDF(o.vertices, o.faces) # (num_vertices, 3) and (num_faces, 3)

# Compute some SDF values (negative outside);
# takes a (num_points, 3) array, converts automatically
origin_sdf = f([0, 0, 0])
print(origin_sdf)
origin_sdf = f([0.0000001, 0, 0])
print(origin_sdf)
sdf_multi_point = f([[0, 0, 0],[1,1,1],[0.1,0.2,0.2]])
print(sdf_multi_point)
# Contains check
origin_contained = f.contains([0, 0, 0])
print(origin_contained)
# Misc: nearest neighbor
origin_nn = f.nn([0, 0, 0])

# Misc: uniform surface point sampling
random_surface_points = f.sample_surface(10000)

# Misc: surface area
the_surface_area = f.surface_area

# All the functions also support an additional argument 'n_threads=<int>' to specify number of threads.
# by default we use min(32, n_cpus)