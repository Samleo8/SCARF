import trimesh

if __name__ == '__main__':
    import sys, os

    expname = sys.argv[1] if len(sys.argv) > 1 else 'train_DTU_2L_32H'
    scan = sys.argv[2] if len(sys.argv) > 2 else '23'

    mesh_path = os.path.join('logs', f'render_{expname}', '3d_mesh',
                             f'mesh_colored_scan{scan}.obj')

    if not os.path.exists(mesh_path):
        print("Mesh file not found:", mesh_path)
        sys.exit(1)

    print("Loading 3D reconstruction for experiment", expname, "scan", scan, end='... ', flush=True)
    mesh = trimesh.load_mesh(mesh_path)
    print("Done!")

    print("Displaying 3D reconstruction using trimesh and pyglet. Please be patient...")
    mesh.show()