import jax.experimental.sparse
import jax.numpy as jnp

from jaxvox._jaxvox import VoxGrid, VoxList

if __name__ == '__main__':
    import open3d as o3d
    import json
    pcd = o3d.io.read_point_cloud("./data/0/pcd.pcd")

    with open("./data/0/other_info.json", "r") as f:
        other_info = json.loads(f.read())

    o3d_voxelgrid_from_point_cloud = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.02)

    def test_o3d_io():
        def compare_grids(o3d_grid1, o3d_grid2):
            voxlist1 = VoxList.from_open3d(o3d_grid1).to_voxelgrid()
            voxlist2 = VoxList.from_open3d(o3d_grid2).to_voxelgrid()
            return jnp.all(voxlist1.grid == voxlist2.grid)

        voxlist = VoxList.from_open3d(o3d_voxelgrid_from_point_cloud)

        new_voxel_grid = voxlist.to_open3d()

        voxlist = VoxList.from_open3d(new_voxel_grid)

        voxgrid = voxlist.to_voxelgrid()

        og_voxel_grid = voxgrid.to_open3d()

        assert compare_grids(og_voxel_grid, new_voxel_grid)

    def test_display():
        voxgrid, attr_mapping = VoxGrid.from_open3d(o3d_voxelgrid_from_point_cloud, return_attrmanager=True)

        attr_mapping.set_default_value((255,0,0))

        def temp(voxgrid):
            #inner_grid = voxgrid.grid
            voxgrid = voxgrid.at[:,0,:].set(1)
            voxgrid.at[:,0,:].get()
            return voxgrid
        voxgrid = jax.jit(temp)(voxgrid)

        for path in other_info["path_pts"]:
            voxgrid = voxgrid.set_point(path)

        #voxgrid = voxgrid.add_voxel()

        attr_mapping = attr_mapping #plt.get_cmap("gist_rainbow") # values to try: attr_mapping, None, and a colormap

        voxgrid.display_as_o3d(attr_mapping)

    def test_display_voxlist():
        voxlist, attr_mapping = VoxList.from_open3d(o3d_voxelgrid_from_point_cloud, return_attrmanager=True)

        #def temp(voxlist):
        #    #inner_grid = voxgrid.grid
        #    voxlist = voxlist.at[:,0,:].set(1)
        #    voxlist.at[:,0,:].get()
        #    return voxlist.to_voxelgrid()
        #voxgrid = jax.jit(temp)(voxlist)
        voxgrid = voxlist.to_voxelgrid()


        #voxgrid = voxgrid.add_voxel()

        attr_mapping = attr_mapping #plt.get_cmap("gist_rainbow") # values to try: attr_mapping, None, and a colormap

        voxgrid.display_as_o3d(attr_mapping)

    def test_some_io():
        voxlist = VoxList.build_from_bounds(
            minbound=jnp.array([-0.5, 1.0, -2.5]),
            maxbound=jnp.array([1.0, 10.0, 0.0]),
            voxel_size=0.05,
        )

        point = jnp.array([0.0, 5.0, -1.0])
        voxel_indices_1 = voxlist.point_to_voxel(point)
        #print(voxel_indices_1)  # Output: [10, 80, 30] (example)

        point = voxlist.minbound - 1
        voxel_indices_2 = voxlist.point_to_voxel(point)
        #print(voxel_indices_2)  # Output: [10, 80, 30] (example)

        voxels = jnp.vstack([voxel_indices_1, voxel_indices_2])
        voxlist = voxlist.set_voxel(voxels, jnp.array([2, 3]))

        assert jnp.all(voxlist.is_voxel_set(voxels) == jnp.array([2, -1]))
        assert jnp.all(voxlist.is_point_set(voxlist.minbound - 1) == -1)

        voxlist = voxlist.set_voxel(voxels, jnp.array([23, 3]))
        assert jnp.all(voxlist.is_voxel_set(voxels) == jnp.array([23, -1]))

        #voxlist = voxlist.deduplicate()


        voxel_grid = voxlist.to_voxelgrid()
        # exit()

        #voxel_grid = voxel_grid.to_voxellist(10)

        assert jnp.all(voxel_grid.is_voxel_set(jnp.array([11, 81, 291])) == -1)

        #print())  # should be -1

        # voxel_grid = voxel_grid.to_voxelgrid()

        voxel_grid = voxel_grid.set_point(jnp.array([0.7, 7.0, -1.0]), jnp.array([28]))

        # todo how do i ensure
        assert jnp.all(voxel_grid.is_voxel_set(voxel_indices_1) == 23)

        voxel_grid = voxel_grid.del_voxel(voxel_indices_1)
        assert jnp.all(voxel_grid.is_voxel_set(voxel_indices_1) == 0)
        assert jnp.all(voxel_grid.to_voxellist().is_voxel_set(voxel_indices_1) == 0)    # asserts that voxellist returns 0 when stuff is not set

        assert jnp.all(voxel_grid.is_point_set(jnp.array([0.7, 7.0, -1.0])) == 28)

        voxellist = voxel_grid.to_voxellist()
        voxellist = voxellist.del_point(jnp.array([0.7, 7.0, -1.0]))
        assert jnp.all(voxellist.is_point_set(jnp.array([0.7, 7.0, -1.0])) == 0)

        voxel_grid = voxel_grid.del_point(jnp.array([0.7, 7.0, -1.0]))

        #print(voxel_grid.voxel_to_point(voxel_indices_1))
        #print(voxel_grid.voxel_to_8points(voxel_indices_1))

    test_some_io()
    test_o3d_io()
    test_display_voxlist()
    test_display()


    exit()
    """



    

    import open3d as o3d
    o3d_vox = voxel_grid.to_open3d()

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(o3d_vox)
    visualizer.poll_events()
    visualizer.update_renderer()

    view_control = visualizer.get_view_control()
    view_control.set_front([1, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_lookat([0, 0, 0])
    visualizer.run()
    i=0
    """